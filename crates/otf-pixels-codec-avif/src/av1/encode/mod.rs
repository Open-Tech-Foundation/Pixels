//! AV1 key-frame encoding for AVIF stills.

#![allow(
    dead_code,
    reason = "the encoder is assembled in stages; its driver lands next and \
              uses every piece here"
)]

pub(crate) mod headers;
