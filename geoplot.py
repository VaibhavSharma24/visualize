"""
Geospatial visualization module for AgentTorch simulations.

Generates 3D time-series maps using Cesium JS by converting simulation states 
into animated GeoJSON visualizations. Supports both color gradients and size-based
representations of agent properties.

Example Usage:
-------------
from agent_torch.visualize import GeoPlot

engine = GeoPlot(config, {
    "cesium_token": "your_token_here",
    "step_time": 3600,
    "coordinates": "agents/coordinates/path",
    "feature": "agents/property/path",
    "visualization_type": "color"
})

for _ in range(num_episodes):
    runner.step(steps)
    engine.render(runner.state_trajectory)
"""

import re
import json
import pandas as pd
import numpy as np
from string import Template
from agent_torch.core.helpers import get_by_path

# Cesium HTML template with dynamic placeholders using $ notation
geoplot_template = """
<!doctype html>
<html lang="en">
	<head>
		<!-- Cesium JS dependencies and basic styling -->
		<meta charset="UTF-8" />
		<meta name="viewport" content="width=device-width, initial-scale=1.0" />
		<title>Cesium Time-Series Heatmap Visualization</title>
		<script src="https://cesium.com/downloads/cesiumjs/releases/1.95/Build/Cesium/Cesium.js"></script>
		<link href="https://cesium.com/downloads/cesiumjs/releases/1.95/Build/Cesium/Widgets/widgets.css" rel="stylesheet" />
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
			// Cesium initialization and configuration
			Cesium.Ion.defaultAccessToken = '$accessToken'
			const viewer = new Cesium.Viewer('cesiumContainer')

			// Color interpolation for value visualization
			function interpolateColor(color1, color2, factor) {
				const result = new Cesium.Color()
				result.red = color1.red + factor * (color2.red - color1.red)
				result.green = color1.green + factor * (color2.green - color1.green)
				result.blue = color1.blue + factor * (color2.blue - color1.blue)
				result.alpha = '$visualType' == 'size' ? 0.2 : 
					color1.alpha + factor * (color2.alpha - color1.alpha)
				return result
			}

			// Value-to-color mapping with normalization
			function getColor(value, min, max) {
				const factor = (value - min) / (max - min)
				return interpolateColor(Cesium.Color.BLUE, Cesium.Color.RED, factor)
			}

			// Size scaling for size-based visualization
			function getPixelSize(value, min, max) {
				const factor = (value - min) / (max - min)
				return 100 * (1 + factor)
			}

			// Process GeoJSON data into time-series format
			function processTimeSeriesData(geoJsonData) {
				const timeSeriesMap = new Map()
				let minValue = Infinity
				let maxValue = -Infinity

				geoJsonData.features.forEach((feature) => {
					const id = feature.properties.id
					const time = Cesium.JulianDate.fromIso8601(feature.properties.time)
					const value = feature.properties.value
					const coordinates = feature.geometry.coordinates

					if (!timeSeriesMap.has(id)) timeSeriesMap.set(id, [])
					timeSeriesMap.get(id).push({ time, value, coordinates })

					minValue = Math.min(minValue, value)
					maxValue = Math.max(maxValue, value)
				})

				return { timeSeriesMap, minValue, maxValue }
			}

			// Create Cesium entities for visualization
			function createTimeSeriesEntities(timeSeriesData, startTime, stopTime) {
				const dataSource = new Cesium.CustomDataSource('AgentTorch Simulation')

				for (const [id, timeSeries] of timeSeriesData.timeSeriesMap) {
					const entity = new Cesium.Entity({
						id: id,
						availability: new Cesium.TimeIntervalCollection([
							new Cesium.TimeInterval({ start: startTime, stop: stopTime }),
						]),
						position: new Cesium.SampledPositionProperty(),
						point: {
							pixelSize: '$visualType' == 'size' ? new Cesium.SampledProperty(Number) : 10,
							color: new Cesium.SampledProperty(Cesium.Color),
						},
						properties: { value: new Cesium.SampledProperty(Number) },
					})

					timeSeries.forEach(({ time, value, coordinates }) => {
						const position = Cesium.Cartesian3.fromDegrees(coordinates[0], coordinates[1])
						entity.position.addSample(time, position)
						entity.properties.value.addSample(time, value)
						entity.point.color.addSample(time, getColor(value, timeSeriesData.minValue, timeSeriesData.maxValue))

						if ('$visualType' == 'size') {
							entity.point.pixelSize.addSample(time, 
								getPixelSize(value, timeSeriesData.minValue, timeSeriesData.maxValue))
						}
					})
					dataSource.entities.add(entity)
				}
				return dataSource
			}

			// Initialize timeline and load data
			const start = Cesium.JulianDate.fromIso8601('$startTime')
			const stop = Cesium.JulianDate.fromIso8601('$stopTime')
			viewer.clock.startTime = start.clone()
			viewer.clock.stopTime = stop.clone()
			viewer.clock.currentTime = start.clone()
			viewer.clock.clockRange = Cesium.ClockRange.LOOP_STOP
			viewer.clock.multiplier = 3600 // 1 hour per second
			viewer.timeline.zoomTo(start, stop)

			// Load all GeoJSON datasets
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
    """Access nested state properties using path notation
    
    Args:
        state: Hierarchical simulation state dictionary
        var: Path string using '/' separators (e.g., 'agents/position')
    
    Returns:
        Value at specified path or None if not found
    """
    return get_by_path(state, re.split("/", var))

class GeoPlot:
    """Main class for generating geospatial visualizations from simulation states
    
    Attributes:
        config: Simulation configuration dictionary
        cesium_token: Cesium Ion API access token
        step_time: Simulation time per step in seconds
        entity_position: State path to agent coordinates
        entity_property: State path to visualized property
        visualization_type: 'color' or 'size' encoding
    """

    def __init__(self, config, options):
        """Initialize visualization engine with configuration
        
        Args:
            config: Simulation configuration dictionary
            options: Visualization parameters including:
                - cesium_token: Cesium API token
                - step_time: Seconds per simulation step
                - coordinates: Path to agent coordinates
                - feature: Path to visualized property
                - visualization_type: Visual encoding type
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
        """Generate visualization files from simulation states
        
        Process Flow:
        1. Extract coordinates and values from state trajectory
        2. Generate temporal sequence for visualization
        3. Convert to GeoJSON format
        4. Create HTML visualization with embedded data
        
        Args:
            state_trajectory: List of simulation states over time
        """
        coords, values = [], []
        sim_name = self.config["simulation_metadata"]["name"]
        geodata_path = f"{sim_name}.geojson"
        geoplot_path = f"{sim_name}.html"

        # Process each episode's final state
        for i in range(len(state_trajectory) - 1):
            final_state = state_trajectory[i][-1]
            
            # Extract coordinates (assumed constant across episodes)
            coords = np.array(read_var(final_state, self.entity_position)).tolist()
            
            # Collect evolving property values
            values.append(np.array(read_var(final_state, self.entity_property)).flatten().tolist())

        # Generate simulation timeline
        start_time = pd.Timestamp.utcnow()
        total_steps = (
            self.config["simulation_metadata"]["num_episodes"]
            * self.config["simulation_metadata"]["num_steps_per_episode"]
        )
        timestamps = [start_time + pd.Timedelta(seconds=i*self.step_time) 
                     for i in range(total_steps)]

        # Convert to GeoJSON format
        geojsons = []
        for idx, coord in enumerate(coords):
            features = [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        # GeoJSON requires [longitude, latitude] order
                        "coordinates": [coord[1], coord[0]]  
                    },
                    "properties": {
                        "value": value_list[idx],
                        "time": time.isoformat(),
                    }
                } for time, value_list in zip(timestamps, values)
            ]
            geojsons.append({"type": "FeatureCollection", "features": features})

        # Write output files
        with open(geodata_path, "w", encoding="utf-8") as f:
            json.dump(geojsons, f, ensure_ascii=False, indent=2)

        # Generate visualization HTML
        with open(geoplot_path, "w", encoding="utf-8") as f:
            f.write(Template(geoplot_template).substitute({
                "accessToken": self.cesium_token,
                "startTime": timestamps[0].isoformat(),
                "stopTime": timestamps[-1].isoformat(),
                "data": json.dumps(geojsons),
                "visualType": self.visualization_type,
            }))