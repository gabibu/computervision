from typing import Optional

import torch


def get_image_coordinates(sidelen: int, image_id: Optional[int], rgb=True):
    tensors_2d = 2 * [torch.linspace(-1, 1, steps=sidelen)]
    mgrid = torch.stack(torch.meshgrid(*tensors_2d), dim=-1)

    if rgb:
        r_tensor = torch.full((sidelen, sidelen), fill_value=-1.0).reshape(sidelen, sidelen, 1)
        g_tensor = torch.full((sidelen, sidelen), fill_value=0.0).reshape(sidelen, sidelen, 1)
        b_tensor = torch.full((sidelen, sidelen), fill_value=1.0).reshape(sidelen, sidelen, 1)

        mgrid_r = torch.cat((mgrid, r_tensor), 2)
        mgrid_g = torch.cat((mgrid, g_tensor), 2)
        mgrid_b = torch.cat((mgrid, b_tensor), 2)

        mgrid = torch.concat((mgrid_r, mgrid_g, mgrid_b), dim=0).reshape(-1, 3)

    if image_id is not None:
        image_id_tensor = torch.full((mgrid.shape[0],), fill_value=image_id).reshape(-1, 1)
        mgrid = torch.cat((mgrid, image_id_tensor), 1)

    return mgrid
