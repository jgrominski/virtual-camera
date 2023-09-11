# Virtual Camera
Basic virtual camera implementation using the MVP (Model View Projection) method. A scene with eight cubes is rendered, and then can be viewed via the controllable virtual camera. The camera can be moved around, rotated, and zoomed in/out.

## Variants
The project was implemented in two variants:
* `virtual_camera.py` - objects in the scene are rendered as wireframe models
* `virtual_camera_solid.py` - objects in the scene are rendered as solid models

## Usage
Launch the desired variant of the project, and explore the scene by controlling the camera.

### Camera controls:
* `W/S` `A/D` - move forwards/backwards and left/right
* `SPACE/LSHIFT` - move up/down
* `E/Q` - zoom in/out
* `I/K` `J/L` `U/O` - rotate the camera along all three axes
* `BACKSPACE` - reset camera position
