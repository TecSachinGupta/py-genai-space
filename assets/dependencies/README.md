# External Libraries

Store external libraries, JARs, and dependency files here.

## Structure

- `spark-connectors/`: Spark connector JARs
- `custom-libs/`: Your custom libraries and packages

## Usage

Reference libraries in your code:

```python
# Add JAR to Spark session
spark.sparkContext.addPyFile("assets/libs/custom-library.zip")
```

## Guidelines

- Document the purpose and version of each library
- Keep libraries up to date
- Consider using package managers when possible
