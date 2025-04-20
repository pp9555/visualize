"""
geoplot.py
----------

This visualization renders a 3-D plot of the data given the state
trajectory of a simulation, and the path of the property to render.

It generates an HTML file that contains code to render the plot using Cesium Ion,
and creates a GeoJSON file from the simulation data.

Example usage:
--------------
from agent_torch.visualize import GeoPlot

# create a simulation
# ...

# create a visualizer with configuration options
engine = GeoPlot(config, {
  "cesium_token": "...",
  "step_time": 3600,                     # time interval (in seconds) between simulation steps
  "coordinates": "agents/consumers/coordinates",
  "feature": "agents/consumers/money_spent",
  "visualization_type": "size",          # example visualization type key
})

# visualize in the runner-loop
for i in range(0, num_episodes):
    runner.step(num_steps_per_episode)
    engine.render(runner.state_trajectory)
"""

import re
import json
import pandas as pd
import numpy as np
from string import Template
from agent_torch.core.helpers import get_by_path

# HTML template for the Cesium visualization.
# This template uses placeholders that will be replaced with runtime data,
# such as the Cesium access token, time stamps, GeoJSON data, and visualization type.
geoplot_template = """
<!doctype html>
<html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta
            name="viewport"
            content="width=device-width, initial-scale=1.0"
        />
        <title>Cesium Time-Series Heatmap Visualization</title>
        <script src="https://cesium.com/downloads/cesiumjs/releases/1.95/Build/Cesium/Cesium.js"></script>
        <link
            href="https://cesium.com/downloads/cesiumjs/releases/1.95/Build/Cesium/Widgets/widgets.css"
            rel="stylesheet"
        />
        <style>
            #cesiumContainer {
                width: 100%;
                height: 100%;
            }
        </style>
    </head>
    <body>
        <div id="cesiumContainer"></div>
        <script>
            // Set the Cesium Ion access token from substituted values
            Cesium.Ion.defaultAccessToken = '$accessToken'

            // Create the Cesium viewer instance
            const viewer = new Cesium.Viewer('cesiumContainer')

            // Linear interpolation between two Cesium colors based on a factor (0 to 1)
            function interpolateColor(color1, color2, factor) {
                const result = new Cesium.Color()
                result.red = color1.red + factor * (color2.red - color1.red)
                result.green = color1.green + factor * (color2.green - color1.green)
                result.blue = color1.blue + factor * (color2.blue - color1.blue)
                // If visualization type is 'size', use a constant alpha; otherwise, interpolate
                result.alpha = '$visualType' == 'size' ? 0.2 :
                    color1.alpha + factor * (color2.alpha - color1.alpha)
                return result
            }

            // Get color for a given value by mapping it between min and max values
            function getColor(value, min, max) {
                const factor = (value - min) / (max - min)
                return interpolateColor(
                    Cesium.Color.BLUE,
                    Cesium.Color.RED,
                    factor
                )
            }

            // Get pixel size for a given value, with linear scaling based on min and max
            function getPixelSize(value, min, max) {
                const factor = (value - min) / (max - min)
                return 100 * (1 + factor)
            }

            // Processes a GeoJSON time series data collection into a map keyed by feature id.
            // Also determines the minimum and maximum property values in the dataset.
            function processTimeSeriesData(geoJsonData) {
                const timeSeriesMap = new Map()
                let minValue = Infinity
                let maxValue = -Infinity

                geoJsonData.features.forEach((feature) => {
                    const id = feature.properties.id
                    // Convert ISO 8601 formatted time to Cesium JulianDate
                    const time = Cesium.JulianDate.fromIso8601(feature.properties.time)
                    const value = feature.properties.value
                    const coordinates = feature.geometry.coordinates

                    // Group time series data based on entity id
                    if (!timeSeriesMap.has(id)) {
                        timeSeriesMap.set(id, [])
                    }
                    timeSeriesMap.get(id).push({ time, value, coordinates })

                    // Update min/max value for scaling
                    minValue = Math.min(minValue, value)
                    maxValue = Math.max(maxValue, value)
                })

                return { timeSeriesMap, minValue, maxValue }
            }

            // Creates Cesium entities from processed time series data.
            // Each entity gets sampled properties for position, point color and optional pixel size.
            function createTimeSeriesEntities(timeSeriesData, startTime, stopTime) {
                const dataSource = new Cesium.CustomDataSource('AgentTorch Simulation')

                for (const [id, timeSeries] of timeSeriesData.timeSeriesMap) {
                    const entity = new Cesium.Entity({
                        id: id,
                        // Define availability of the entity over the entire simulation time
                        availability: new Cesium.TimeIntervalCollection([
                            new Cesium.TimeInterval({
                                start: startTime,
                                stop: stopTime,
                            }),
                        ]),
                        position: new Cesium.SampledPositionProperty(),
                        point: {
                            // Use a sampled property for pixelSize only if visualization type is 'size'
                            pixelSize: '$visualType' == 'size' ? new Cesium.SampledProperty(Number) : 10,
                            color: new Cesium.SampledProperty(Cesium.Color),
                        },
                        properties: {
                            value: new Cesium.SampledProperty(Number),
                        },
                    })

                    // Populate time-sampled data for each entity
                    timeSeries.forEach(({ time, value, coordinates }) => {
                        const position = Cesium.Cartesian3.fromDegrees(
                            coordinates[0],
                            coordinates[1]
                        )
                        entity.position.addSample(time, position)
                        entity.properties.value.addSample(time, value)
                        entity.point.color.addSample(
                            time,
                            getColor(value, timeSeriesData.minValue, timeSeriesData.maxValue)
                        )

                        // If type is 'size', also sample pixel size dynamically
                        if ('$visualType' == 'size') {
                          entity.point.pixelSize.addSample(
                            time,
                            getPixelSize(value, timeSeriesData.minValue, timeSeriesData.maxValue)
                        )
                        }
                    })

                    dataSource.entities.add(entity)
                }

                return dataSource
            }

            // geoJsons is injected as a JSON string of time-series GeoJSON data.
            const geoJsons = $data

            // Parse the start and stop times from ISO8601 strings injected via template
            const start = Cesium.JulianDate.fromIso8601('$startTime')
            const stop = Cesium.JulianDate.fromIso8601('$stopTime')

            // Configure the Cesium clock and timeline for simulation playback
            viewer.clock.startTime = start.clone()
            viewer.clock.stopTime = stop.clone()
            viewer.clock.currentTime = start.clone()
            viewer.clock.clockRange = Cesium.ClockRange.LOOP_STOP
            viewer.clock.multiplier = 3600 // 1 hour per second simulation speed

            viewer.timeline.zoomTo(start, stop)

            // Process and render each GeoJSON dataset
            for (const geoJsonData of geoJsons) {
                const timeSeriesData = processTimeSeriesData(geoJsonData)
                const dataSource = createTimeSeriesEntities(timeSeriesData, start, stop)
                viewer.dataSources.add(dataSource)
                viewer.zoomTo(dataSource)
            }
        </script>
    </body>
</html>
"""

def read_var(state, var):
    """
    Retrieve a variable from the state dictionary by using a '/' separated path.
    For example, a path like "agents/consumers/coordinates" is split into keys.
    """
    return get_by_path(state, re.split("/", var))


class GeoPlot:
    """
    GeoPlot generates a Cesium-based visualization from simulation data.
    It processes the simulation state trajectory, generates GeoJSON data,
    and creates an HTML file that visualizes the time-series data.
    """
    def __init__(self, config, options):
        """
        Initialize the GeoPlot visualizer with simulation configuration and options.

        Parameters:
        - config: Dictionary containing simulation metadata (e.g., simulation name).
        - options: Dictionary containing visualization options such as:
            - cesium_token: Cesium Ion access token for rendering.
            - step_time: Time interval (in seconds) between simulation steps.
            - coordinates: Path in the state leading to agent coordinates.
            - feature: Path in the state leading to the property being visualized.
            - visualization_type: Determines whether the visualization varies by size or color.
        """
        self.config = config
        (
            self.cesium_token,
            self.step_time,
            self.entity_position,
            self.entity_property,
            self.visualization_type,
        ) = (
            options["cesium_token"],
            options["step_time"],
            options["coordinates"],
            options["feature"],
            options["visualization_type"],
        )

    def render(self, state_trajectory):
        """
        Process the simulation state trajectory to generate GeoJSON files and an HTML visualization.
        
        Parameters:
        - state_trajectory: A list of simulation states over time.
          Each element is expected to be a nested structure where the final state
          contains the coordinate and property data for visualization.
        """
        coords, values = [], []
        # Retrieve simulation name to construct output file paths.
        name = self.config["simulation_metadata"]["name"]
        geodata_path, geoplot_path = f"{name}.geojson", f"{name}.html"

        # Iterate through the simulation state trajectory.
        # For each simulation step, extract coordinates and corresponding property values.
        for i in range(0, len(state_trajectory) - 1):
            final_state = state_trajectory[i][-1]

            # Extract coordinates using the provided path (entity_position)
            coords = np.array(read_var(final_state, self.entity_position)).tolist()
            # Extract property values using the provided path (entity_property)
            values.append(
                np.array(read_var(final_state, self.entity_property)).flatten().tolist()
            )

        # Determine the simulation start time using the current UTC timestamp.
        start_time = pd.Timestamp.utcnow()
        # Create a list of timestamps corresponding to each simulation step.
        timestamps = [
            start_time + pd.Timedelta(seconds=i * self.step_time)
            for i in range(
                self.config["simulation_metadata"]["num_episodes"]
                * self.config["simulation_metadata"]["num_steps_per_episode"]
            )
        ]

        geojsons = []
        # Iterate over each coordinate (each agent or point).
        for i, coord in enumerate(coords):
            features = []
            # For each timestamp, create a GeoJSON feature with the corresponding property value.
            for time, value_list in zip(timestamps, values):
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {
                            # GeoJSON expects coordinates in [longitude, latitude] order.
                            "type": "Point",
                            "coordinates": [coord[1], coord[0]],
                        },
                        "properties": {
                            "value": value_list[i],
                            "time": time.isoformat(),
                        },
                    }
                )
            # Aggregate features into a GeoJSON FeatureCollection.
            geojsons.append({"type": "FeatureCollection", "features": features})

        # Write the GeoJSON data to a file for use by the Cesium visualization.
        with open(geodata_path, "w", encoding="utf-8") as f:
            json.dump(geojsons, f, ensure_ascii=False, indent=2)

        # Use Python's Template engine to substitute runtime parameters into the HTML template.
        tmpl = Template(geoplot_template)
        with open(geoplot_path, "w", encoding="utf-8") as f:
            f.write(
                tmpl.substitute(
                    {
                        "accessToken": self.cesium_token,
                        "startTime": timestamps[0].isoformat(),
                        "stopTime": timestamps[-1].isoformat(),
                        "data": json.dumps(geojsons),
                        "visualType": self.visualization_type,
                    }
                )
            )