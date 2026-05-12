from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from common import ensure_parent


GRAPH_TEMPLATE = """<graph id="Sentinel1_GRD_Cal_TC">
  <version>1.0</version>
  <node id="Read">
    <operator>Read</operator>
    <parameters>
      <file>{input_path}</file>
    </parameters>
  </node>
  <node id="Apply-Orbit-File">
    <operator>Apply-Orbit-File</operator>
    <sources>
      <sourceProduct refid="Read"/>
    </sources>
    <parameters>
      <orbitType>Sentinel Precise (Auto Download)</orbitType>
      <polyDegree>3</polyDegree>
      <continueOnFail>true</continueOnFail>
    </parameters>
  </node>
  <node id="Remove-GRD-Border-Noise">
    <operator>Remove-GRD-Border-Noise</operator>
    <sources>
      <sourceProduct refid="Apply-Orbit-File"/>
    </sources>
  </node>
  <node id="Calibration">
    <operator>Calibration</operator>
    <sources>
      <sourceProduct refid="Remove-GRD-Border-Noise"/>
    </sources>
    <parameters>
      <outputSigmaBand>true</outputSigmaBand>
      <selectedPolarisations>{polarizations}</selectedPolarisations>
    </parameters>
  </node>
  {subset_node}
  {speckle_node}
  <node id="Terrain-Correction">
    <operator>Terrain-Correction</operator>
    <sources>
      <sourceProduct refid="{terrain_source}"/>
    </sources>
    <parameters>
      <demName>{dem_name}</demName>
      <pixelSpacingInMeter>{pixel_spacing}</pixelSpacingInMeter>
      <mapProjection>{map_projection}</mapProjection>
      <imgResamplingMethod>BILINEAR_INTERPOLATION</imgResamplingMethod>
      <demResamplingMethod>BILINEAR_INTERPOLATION</demResamplingMethod>
      <saveSelectedSourceBand>true</saveSelectedSourceBand>
    </parameters>
  </node>
  <node id="Write">
    <operator>Write</operator>
    <sources>
      <sourceProduct refid="Terrain-Correction"/>
    </sources>
    <parameters>
      <file>{output_path}</file>
      <formatName>GeoTIFF</formatName>
    </parameters>
  </node>
</graph>
"""


SPECKLE_NODE = """<node id="Speckle-Filter">
    <operator>Speckle-Filter</operator>
    <sources>
      <sourceProduct refid="{speckle_source}"/>
    </sources>
    <parameters>
      <filter>Lee Sigma</filter>
      <filterSizeX>5</filterSizeX>
      <filterSizeY>5</filterSizeY>
    </parameters>
  </node>"""


SUBSET_NODE = """<node id="Subset">
    <operator>Subset</operator>
    <sources>
      <sourceProduct refid="Calibration"/>
    </sources>
    <parameters>
      <geoRegion>{geo_region}</geoRegion>
      <copyMetadata>true</copyMetadata>
    </parameters>
  </node>"""


def build_graph(args: argparse.Namespace) -> str:
    use_speckle = args.speckle
    subset_node = SUBSET_NODE.format(geo_region=args.geo_region) if args.geo_region else ""
    calibrated_source = "Subset" if args.geo_region else "Calibration"
    return GRAPH_TEMPLATE.format(
        input_path=Path(args.input).resolve().as_posix(),
        output_path=Path(args.output).resolve().as_posix(),
        polarizations=args.polarizations,
        pixel_spacing=args.pixel_spacing,
        dem_name=args.dem_name,
        map_projection=args.map_projection,
        subset_node=subset_node,
        speckle_node=SPECKLE_NODE.format(speckle_source=calibrated_source) if use_speckle else "",
        terrain_source="Speckle-Filter" if use_speckle else calibrated_source,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess Sentinel-1 GRD with SNAP GPT: orbit, border-noise removal, calibration, optional speckle filtering, terrain correction, GeoTIFF export."
    )
    parser.add_argument("--input", required=True, help="Sentinel-1 GRD .zip or .SAFE path")
    parser.add_argument("--output", default="data/processed/sar_10m.tif", help="Output GeoTIFF")
    parser.add_argument("--gpt", default="gpt", help="SNAP GPT executable path")
    parser.add_argument("--polarizations", default="VV,VH", help="Polarizations, e.g. VV or VV,VH")
    parser.add_argument("--pixel-spacing", type=float, default=10.0)
    parser.add_argument("--dem-name", default="SRTM 1Sec HGT")
    parser.add_argument("--map-projection", default="EPSG:32648")
    parser.add_argument(
        "--geo-region",
        help="Optional WGS84 WKT polygon for AOI subsetting before terrain correction. Example: POLYGON((104.02 30.32,104.20 30.32,104.20 30.48,104.02 30.48,104.02 30.32))",
    )
    parser.add_argument("--speckle", action="store_true", help="Enable Lee Sigma speckle filtering")
    parser.add_argument("--write-graph-only", action="store_true", help="Only write the SNAP graph XML next to output")
    args = parser.parse_args()

    output = ensure_parent(args.output)
    graph_path = output.with_suffix(".snap.graph.xml")
    graph_path.write_text(build_graph(args), encoding="utf-8")

    if args.write_graph_only:
        print(f"Wrote SNAP graph: {graph_path}")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp_graph = Path(tmp) / graph_path.name
        tmp_graph.write_text(graph_path.read_text(encoding="utf-8"), encoding="utf-8")
        subprocess.run([args.gpt, str(tmp_graph)], check=True)

    print(f"Wrote SAR GeoTIFF: {output}")


if __name__ == "__main__":
    main()
