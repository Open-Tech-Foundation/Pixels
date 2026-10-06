# ADR-0015: ICC profiles — preserve always, convert matrix/TRC to sRGB

Status: Accepted · 2026-10-07

## Context
v1 assumed every pixel was sRGB and ignored embedded profiles (SPEC §Pixel
formats), leaving "ICC color management" as a v2 roadmap item. That
assumption is false for a large share of what an image API is fed: phones
shoot Display P3, cameras and editors export Adobe RGB or ProPhoto, and HDR
pipelines tag BT.2020. Treated as sRGB, a P3 photo comes out dull; a ProPhoto
export, muddy. Worse, re-encoding without the profile throws away the only
information that would let a browser show it right, so every pipeline run
silently damaged colour.

Two separable needs hide in "ICC support":

1. **Preserve** — carry the profile from input to output, so nothing is lost
   even when nothing is converted.
2. **Convert** — bring the pixels into sRGB, the space every op here and most
   consumers assume, as sharp does by default.

A full colour management module (lcms2's scope: LUT-based `A2B`/`B2A`
profiles, CMYK, Lab PCS, every rendering intent, black point compensation) is
a project of its own. The profiles actually embedded in RGB and grey images
are overwhelmingly **matrix/TRC**: three colorants and three tone curves,
which a few hundred lines convert exactly.

## Decision
- Every decoder reports an embedded profile (`Decoder::icc_profile`) and
  every encoder whose format has a place for one writes it
  (`Encoder::set_icc_profile`): PNG `iCCP`, JPEG `APP2`, WebP `ICCP`, TIFF
  tag 34675, AVIF `colr`. The facade carries the profile on `Image`.
- `ToSrgb` (otf-pixels-ops) converts matrix/TRC RGB and grey `kTRC`
  profiles: `curv` (gamma or table) and all five `para` curve types,
  relative colorimetric through the D50 PCS with sRGB's colorants built as
  lcms2 builds them, out-of-gamut colours clipped. A profile that is sRGB in
  all but name is recognised and only dropped.
- Opening converts by default (`OpenOptions::to_srgb`); off, pixels arrive
  as stored and `Image::to_srgb` converts later. A converted image carries no
  profile; anything not converted keeps its profile, so it is still written
  out and still displayed right by a colour-managed viewer.
- lcms2 is the reference: generated fixtures must match it within one 8-bit
  step, and so do the 33 RGB and grey profiles colord and Ghostscript ship.

## Consequences
- Wide-gamut photos look right after a resize, and a pipeline that does not
  convert no longer strips colour information.
- Converting to sRGB clips: a P3 red outside sRGB becomes sRGB's red. Keeping
  wide gamut means opening with `to_srgb` off and writing the profile back.
- LUT-based profiles, CMYK, Lab-PCS profiles, perceptual/saturation intents
  and black point compensation are not implemented; such profiles pass
  through unconverted. Converting *to* a profile other than sRGB is not
  offered. Each is additive should demand appear.
- GIF and raw output cannot carry a profile and drop it; their pixels should
  be converted first, which the default does.
