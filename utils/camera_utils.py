from pytorch3d.renderer.cameras import camera_position_from_spherical_angles, look_at_view_transform
import torch

class CameraUtils:
    def __init__(self):
        pass

    def generate_spherical_camera_array(self, sphere_radius: int, sample_type:str = "semi-sphere", sample_interval_degree: int = 60, device = "cuda"):
        """
        Generate the camera coordiantes list for spherical sampling. The coordinate system is 
        compatible of pytorch3D coordinate system, which follows the right hand rule the vertical
        axis is y. The axes sequence is x, z, y on clock-wise order.
        
        sphere_radius: define the radius of spherical coordinate system; int
        sample_type: define the sampling type for spherical surface. str "semi-sphere", "sphere"
        sample_interval_degree: define the angle degree of spherical coordinate system. int
        ------------------------------------------------------------------------------------------
        Reture: return the list of camera parameter array. list[tuple(x, y, z, yaw, pitch), ...]
        """
        azimuth_list = range(0, 360, sample_interval_degree)
        
        if sample_type == "semi-sphere":
            elevation_list = range(0, 90, 10)
        else:
            elevation_list = range(-90, 90, 10)
        
        camera_postion_list = []
        for elevation in elevation_list:
            for azimuth in azimuth_list:
                camera_position = camera_position_from_spherical_angles(sphere_radius, elevation, azimuth, device=device)
                camera_postion_list.append(camera_position)
        #camera_postion_list.append(torch.tensor([[0., float(sphere_radius), 0.]], device=device))

        camera_positions = torch.cat(camera_postion_list, 0)

        Rs, Ts = look_at_view_transform(eye=camera_positions)

        return Rs, Ts, camera_positions