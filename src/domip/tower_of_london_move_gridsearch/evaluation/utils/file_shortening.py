"""Create shortened copies of files in a folder.

Each underscore-delimited segment in a filename is shortened to its first
three letters. Segments that are purely numeric stay unchanged. The shortened
copies are written to the same folder as the original files.

This file contains code written by AI.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


ALPHA_SEGMENT = re.compile(r"^([A-Za-z]+)(.*)$")


def shorten_segment(segment: str) -> str:
	"""Shorten one underscore-delimited filename segment."""

	if segment.isdigit():
		return segment

	match = ALPHA_SEGMENT.match(segment)
	if not match:
		return segment

	letters, remainder = match.groups()
	return f"{letters[:3]}{remainder}"


def shorten_filename(filename: str) -> str:
	"""Shorten a filename stem while preserving its extension."""

	path = Path(filename)
	shortened_stem = "_".join(shorten_segment(part) for part in path.stem.split("_"))
	return f"{shortened_stem}{path.suffix}"


def unique_target_path(target_path: Path) -> Path:
	"""Return a non-existing path by adding a numeric suffix if needed."""

	if not target_path.exists():
		return target_path

	counter = 1
	while True:
		candidate = target_path.with_name(
			f"{target_path.stem}_{counter}{target_path.suffix}"
		)
		if not candidate.exists():
			return candidate
		counter += 1


def shorten_files_in_folder(folder: Path) -> list[Path]:
	"""Create shortened copies for every regular file in *folder*."""

	created_files: list[Path] = []
	script_path = Path(__file__).resolve()

	for source_path in folder.iterdir():
		if not source_path.is_file():
			continue
		if source_path.resolve() == script_path:
			continue

		target_name = shorten_filename(source_path.name)
		target_path = unique_target_path(source_path.with_name(target_name))
		shutil.copy2(source_path, target_path)
		created_files.append(target_path)

	return created_files


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Create shortened filename copies in the same folder."
	)
	parser.add_argument(
		"folder",
		nargs="?",
		default=Path(__file__).resolve().parent,
		type=Path,
		help="Folder containing the files to shorten.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	folder = args.folder.resolve()
	created_files = shorten_files_in_folder(folder)

	for created_file in created_files:
		print(created_file)


if __name__ == "__main__":
	main()
