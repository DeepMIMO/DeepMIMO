# DeepMIMO Validation Tests

This folder contains Python test scripts that validate DeepMIMO scenario files. Each test script checks a specific aspect of the uploaded files (e.g., file extensions, structure, parameters) and returns a standardized pass/fail result.

## Test Format

Each test script should:
1. Accept a file path as a command line argument
2. Return a JSON object with the format:
```json
{
    "valid": true|false,
    "error": null|"error message"
}
```

## Running Tests

Individual test:
```bash
python validate_extensions.py /path/to/scenario.zip
```

## Creating New Tests

1. Create a new Python script in this directory
2. Accept a single command line argument (file path)
3. Return JSON in the required format
4. Exit with code 0 on successful execution

## Requirements

- Python 3.x
- zipfile module
- json module
- os module

## Test Guidelines

- Each test should validate one specific aspect
- Tests must be independent of each other
- Clean up any temporary files after execution
- Use clear, descriptive error messages