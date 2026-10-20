#!/usr/bin/env python3
"""Regenerate the metadata fixtures: small JPEG, PNG, WebP and TIFF files that
Pillow writes with an XMP packet, an EXIF block (Make "Pillow", Orientation 3,
ExposureTime 1/125; TIFF holds the first two as tags of its own) and a density, plus PNG text chunks, checked by
crates/otf-pixels/tests/source_metadata.rs. Requires Pillow 11.
"""
import os
from PIL import Image, PngImagePlugin

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "crates", "otf-pixels", "tests", "fixtures", "metadata")
XMP = (b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?><x:xmpmeta xmlns:x="adobe:ns:meta/">'
       b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description '
       b'xmlns:dc="http://purl.org/dc/elements/1.1/" dc:creator="Pillow"/></rdf:RDF></x:xmpmeta><?xpacket end="w"?>')

os.makedirs(OUT, exist_ok=True)
image = Image.new("RGB", (24, 16))
image.putdata([(x * 10, y * 15, 128) for y in range(16) for x in range(24)])
exif = Image.Exif()
exif[0x010F] = "Pillow"
exif[0x0112] = 3
exif.get_ifd(0x8769)[0x829A] = 1 / 125
raw = exif.tobytes()
image.save(os.path.join(OUT, "pillow.jpg"), exif=raw, xmp=XMP, dpi=(240, 240), quality=90, progressive=True)
text = PngImagePlugin.PngInfo()
text.add_text("Title", "Harbour")
text.add_itxt("XML:com.adobe.xmp", XMP.decode())
text.add_text("Comment", "zipped " * 20, zip=True)
image.save(os.path.join(OUT, "pillow.png"), exif=raw, pnginfo=text, dpi=(240, 240))
image.save(os.path.join(OUT, "pillow.webp"), exif=raw, xmp=XMP, lossless=True)
# TIFF keeps EXIF as tags in its own directory.
image.save(os.path.join(OUT, "pillow.tif"), tiffinfo={700: XMP, 271: "Pillow", 274: 3}, dpi=(240, 240), compression="tiff_lzw")
