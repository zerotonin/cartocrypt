"""Click-based command-line interface for CartoCrypt.

« cartocrypt anonymise --key secret.key --bbox ... »
"""

from __future__ import annotations

from pathlib import Path

import click


@click.group()
@click.version_option()
def main() -> None:
    """CartoCrypt — topology-preserving cryptographic map anonymisation."""


@main.command()
@click.option("--output", "-o", type=click.Path(path_type=Path), required=True,
              help="Output key file path.")
def keygen(output: Path) -> None:
    """Generate a new 32-byte symmetric key."""
    from cartocrypt.keygen import generate_key, save_key

    key = generate_key()
    save_key(key, output)
    fingerprint = key[:4].hex()
    click.echo(f"Key generated → {output}  (fingerprint: {fingerprint}...)")


@main.command()
@click.option("--bbox", "-b", type=str, required=True,
              help="Bounding box as 'north,south,east,west'.")
@click.option("--key", "-k", type=click.Path(exists=True, path_type=Path),
              required=True, help="Path to symmetric key file.")
@click.option("--output", "-o", type=click.Path(path_type=Path), required=True,
              help="Output GeoJSON file path.")
@click.option("--network-type", "-n", type=str, default="drive",
              help="OSM network type (drive, walk, bike, all).")
@click.option("--svg/--no-svg", default=True,
              help="Also generate an SVG visualisation.")
def anonymise(
    bbox: str,
    key: Path,
    output: Path,
    network_type: str,
    svg: bool,
) -> None:
    """Anonymise an OSM bounding box and export to GeoJSON."""
    from cartocrypt.canon import attribute_hash, weisfeiler_lehman_hash
    from cartocrypt.export import to_geojson, to_svg
    from cartocrypt.ingest import from_osm, to_labelled_graph
    from cartocrypt.keygen import (
        compute_checksum,
        load_key,
        prf_coordinates_batch,
    )
    from cartocrypt.reembed import reembed

    # Parse bbox
    parts = [float(x.strip()) for x in bbox.split(",")]
    if len(parts) != 4:
        msg = "Bounding box must have exactly 4 values: north,south,east,west"
        raise click.BadParameter(msg)
    bbox_tuple = (parts[0], parts[1], parts[2], parts[3])

    click.echo("Downloading OSM data...")
    raw = from_osm(bbox_tuple, network_type=network_type)

    click.echo("Extracting labelled graph...")
    g, coords, metadata = to_labelled_graph(raw)

    click.echo(f"Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    click.echo("Loading key and generating seed coordinates...")
    k = load_key(key)
    seed = prf_coordinates_batch(k, g.number_of_nodes())

    click.echo("Re-embedding (Tutte + stress majorisation)...")
    anon_coords = reembed(g, coords, seed)

    click.echo("Computing checksum...")
    gh = weisfeiler_lehman_hash(g)
    ah = attribute_hash(g)
    checksum = compute_checksum(k, gh, ah)
    metadata["checksum"] = checksum
    click.echo(f"Checksum: {checksum[:16]}...")

    click.echo(f"Exporting → {output}")
    to_geojson(g, anon_coords, metadata, output)

    if svg:
        svg_path = output.with_suffix(".svg")
        to_svg(g, anon_coords, svg_path)
        click.echo(f"SVG → {svg_path}")

    click.echo("Done.")


@main.command()
@click.option("--key", "-k", type=click.Path(exists=True, path_type=Path),
              required=True, help="Path to symmetric key file.")
@click.option("--checksum", "-c", type=str, required=True,
              help="Expected checksum string.")
@click.option("--geojson", "-g", type=click.Path(exists=True, path_type=Path),
              required=True, help="Path to anonymised GeoJSON.")
def verify(key: Path, checksum: str, geojson: Path) -> None:
    """Verify a checksum against an anonymised dataset."""
    click.echo("Verification not yet fully implemented.")
    click.echo(f"Key: {key}, Checksum: {checksum[:16]}..., File: {geojson}")
