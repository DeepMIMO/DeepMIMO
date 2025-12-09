"""Validation utilities for uploaded ZIP archives in web tests."""

import json
import sys
import zipfile
from pathlib import Path

ALLOWED_EXTENSIONS = {".mat", ".city", ".ter", ".txrx", ".setup"}
MAX_FILES = 20


def validate_zip_contents(zip_path):
    """Check ZIP file contents for allowed count and file extensions."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Get all file names in zip
            files = zip_ref.namelist()

            # Check total number of files (excluding directories)
            actual_files = [f for f in files if not f.endswith("/")]
            if len(actual_files) > MAX_FILES:
                return {
                    "valid": False,
                    "error": f"Too many files in zip. Maximum allowed is {MAX_FILES}",
                }

            # Check each file's extension
            for file in actual_files:
                # Get file extension (lowercase for consistency)
                ext = Path(file).suffix.lower()

                # If file has an extension and it's not in allowed list
                if ext and ext not in ALLOWED_EXTENSIONS:
                    return {
                        "valid": False,
                        "error": f"Invalid file extension found: {ext}",
                    }

            return {
                "valid": True,
                "error": None,
            }

    except zipfile.BadZipFile:
        return {
            "valid": False,
            "error": "Invalid zip file",
        }
    except (OSError, zipfile.LargeZipFile) as e:
        return {
            "valid": False,
            "error": str(e),
        }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            json.dumps(
                {
                    "valid": False,
                    "error": "Invalid arguments",
                },
            ),
        )
        sys.exit(1)

    result = validate_zip_contents(sys.argv[1])
    print(json.dumps(result))
