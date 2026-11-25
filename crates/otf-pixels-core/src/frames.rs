//! One graph, replayed over every frame of an animation (ADR-0020).
//!
//! A pipeline is built once, against the first frame's source. Running it
//! again for the next frame is not the same graph run twice: the scheduler's
//! tile cache is keyed by node, so a second run would be handed the first
//! frame's tiles. [`FrameReplay::graph`] therefore rebuilds the graph for
//! each frame — the same op instances, which are bound to the canvas size
//! every frame shares, under fresh node identities.
//!
//! # What does not change between frames
//!
//! A sub-graph that does not read the animated source — a watermark
//! composited onto every frame, say — is the same for every frame. It is
//! evaluated once, when the replay is built, and every frame's graph shares
//! that one buffer. That is also what makes it safe: such a sub-graph often
//! reads a streaming decoder of its own, which could not be read a second
//! time.

use std::collections::HashMap;
use std::sync::Arc;

use crate::{BufferSource, Image, Node, NodeId, PixelsError, Producer, Result};

/// A template graph and the animated source it replays over.
#[derive(Debug)]
pub struct FrameReplay {
    template: Image,
    /// The template's node for the animated source.
    source: NodeId,
    /// What that node is replaced with in every frame's graph.
    producer: Arc<dyn Producer>,
    /// Sub-graphs that do not read the source, evaluated once.
    statics: HashMap<NodeId, Image>,
}

impl FrameReplay {
    /// Prepare to replay `template` with its node `source` replaced by
    /// `producer`, evaluating once every sub-graph that does not read it.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Graph`] if `template` has no node `source`, or
    /// whatever evaluating a frame-independent sub-graph returns.
    pub fn new(
        template: &Image,
        source: NodeId,
        producer: Arc<dyn Producer>,
        cancel: &crate::Cancel,
    ) -> Result<Self> {
        let mut depends = HashMap::new();
        if !reads(template.node(), source, &mut depends) {
            return Err(PixelsError::graph(
                "the animated source is not part of this pipeline",
            ));
        }
        let mut statics = HashMap::new();
        collect_statics(template, template.node(), &depends, &mut statics, cancel)?;
        Ok(Self {
            template: template.clone(),
            source,
            producer,
            statics,
        })
    }

    /// The template rebuilt for the frame the producer is now on, with fresh
    /// node identities.
    ///
    /// # Errors
    ///
    /// Propagates an op rejecting its inputs, which cannot happen for a
    /// producer of the source's shape.
    pub fn graph(&self) -> Result<Image> {
        rebind(
            &self.template,
            self.source,
            &self.producer,
            &self.statics,
            &mut HashMap::new(),
            self.template.node(),
        )
    }
}

/// `template` with its node `source` replaced by `producer`, every node that
/// reads it rebuilt: the graph of one still frame, for a caller that needs
/// only one and has nothing to evaluate once.
///
/// # Errors
///
/// Returns [`PixelsError::Graph`] if `template` has no node `source`.
pub fn rebind_source(
    template: &Image,
    source: NodeId,
    producer: Arc<dyn Producer>,
) -> Result<Image> {
    if !reads(template.node(), source, &mut HashMap::new()) {
        return Err(PixelsError::graph(
            "the animated source is not part of this pipeline",
        ));
    }
    rebind(
        template,
        source,
        &producer,
        &HashMap::new(),
        &mut HashMap::new(),
        template.node(),
    )
}

/// Whether `node` reads `source`, memoized in `depends`.
fn reads(node: &Arc<Node>, source: NodeId, depends: &mut HashMap<NodeId, bool>) -> bool {
    if let Some(&known) = depends.get(&node.id()) {
        return known;
    }
    // Every input is visited, never short-circuited: `collect_statics`
    // reads this memo, and an input missing from it would be taken for one
    // that does not read the source.
    let mut found = node.id() == source;
    for input in node.inputs() {
        found |= reads(input, source, depends);
    }
    depends.insert(node.id(), found);
    found
}

/// Evaluate every maximal sub-graph under `node` that does not read the
/// source, keyed by its root.
fn collect_statics(
    template: &Image,
    node: &Arc<Node>,
    depends: &HashMap<NodeId, bool>,
    statics: &mut HashMap<NodeId, Image>,
    cancel: &crate::Cancel,
) -> Result<()> {
    for input in node.inputs() {
        if statics.contains_key(&input.id()) {
            continue;
        }
        if depends.get(&input.id()).copied().unwrap_or(false) {
            collect_statics(template, input, depends, statics, cancel)?;
        } else {
            let sub = template.with_node(Arc::clone(input));
            let pixels = crate::evaluate_cancellable(&sub, cancel)?;
            let buffer = BufferSource::new(input.descriptor(), Arc::new(pixels))?;
            statics.insert(
                input.id(),
                Image::from_producer(Arc::new(buffer), template.metadata()?.format),
            );
        }
    }
    Ok(())
}

fn rebind(
    template: &Image,
    source: NodeId,
    producer: &Arc<dyn Producer>,
    statics: &HashMap<NodeId, Image>,
    memo: &mut HashMap<NodeId, Image>,
    node: &Arc<Node>,
) -> Result<Image> {
    if let Some(built) = memo.get(&node.id()).or_else(|| statics.get(&node.id())) {
        return Ok(built.clone());
    }
    let built = if node.id() == source {
        Image::from_producer(Arc::clone(producer), template.metadata()?.format)
    } else if let Some(op) = node.op() {
        let inputs = node
            .inputs()
            .iter()
            .map(|input| rebind(template, source, producer, statics, memo, input))
            .collect::<Result<Vec<_>>>()?;
        Image::combine(&inputs, Arc::clone(op))?
    } else {
        // Another source, not read through `statics` (`rebind_source`): kept
        // as it is.
        template.with_node(Arc::clone(node))
    };
    memo.insert(node.id(), built.clone());
    Ok(built)
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values"
)]
mod tests {
    use super::*;
    use crate::testing::{FrameStubDecoder, SumOp};
    use crate::{DecodedSource, Format, ImageDescriptor, PixelFormat, Scheduler, TileBuf};

    fn canvas() -> ImageDescriptor {
        ImageDescriptor::new(5, 4, PixelFormat::Gray8).unwrap()
    }

    fn pixels(scheduler: &Scheduler, image: &Image) -> Vec<u8> {
        let mut out = TileBuf::for_image(&image.descriptor()).unwrap();
        {
            let mut view = out.as_tile_mut().unwrap();
            scheduler
                .run(image, |region, tile| {
                    crate::copy_region(tile, &mut view, region)
                })
                .unwrap();
        }
        out.into_bytes()
    }

    #[test]
    fn every_frame_gets_its_own_pixels_through_one_long_lived_scheduler() {
        // The scheduler's cache outlives a run; rebuilding is what keeps
        // frame 1 from being served frame 0's tiles.
        let scheduler = Scheduler::with_defaults().unwrap();
        let first = Arc::new(DecodedSource::new(Box::new(FrameStubDecoder::new(
            canvas(),
            3,
        ))));
        let template = Image::from_producer(first, Format::Raw);
        let flat = Image::from_producer(
            Arc::new(
                BufferSource::new(
                    canvas(),
                    Arc::new(
                        TileBuf::from_vec(canvas().region(), PixelFormat::Gray8, vec![100; 20])
                            .unwrap(),
                    ),
                )
                .unwrap(),
            ),
            Format::Raw,
        );
        // frame + 100, through a two-input op whose second input is static.
        let template = Image::combine(&[template, flat], Arc::new(SumOp)).unwrap();
        let source_id = template.node().inputs()[0].id();

        let live = Arc::new(DecodedSource::new(Box::new(FrameStubDecoder::new(
            canvas(),
            3,
        ))));
        let replay = FrameReplay::new(
            &template,
            source_id,
            Arc::clone(&live) as Arc<dyn Producer>,
            &crate::Cancel::never(),
        )
        .unwrap();
        assert_eq!(
            replay.statics.len(),
            1,
            "the constant input is evaluated once"
        );
        for k in 0..3_u8 {
            if k > 0 {
                assert_eq!(live.advance_frame().unwrap().unwrap().index, u32::from(k));
            }
            let graph = replay.graph().unwrap();
            assert_ne!(graph.node().id(), template.node().id());
            assert!(
                pixels(&scheduler, &graph).iter().all(|&v| v == 100 + k),
                "frame {k}"
            );
        }
        assert_eq!(live.advance_frame().unwrap(), None);
    }

    #[test]
    fn a_template_without_the_source_is_refused() {
        let a = Image::from_producer(
            Arc::new(DecodedSource::new(Box::new(FrameStubDecoder::new(
                canvas(),
                2,
            )))),
            Format::Raw,
        );
        let b = Arc::new(DecodedSource::new(Box::new(FrameStubDecoder::new(
            canvas(),
            2,
        ))));
        let unrelated = Image::from_producer(Arc::clone(&b) as Arc<dyn Producer>, Format::Raw);
        assert!(
            FrameReplay::new(
                &a,
                unrelated.node().id(),
                b.clone(),
                &crate::Cancel::never()
            )
            .is_err()
        );
        assert!(rebind_source(&a, unrelated.node().id(), b).is_err());
    }
}
