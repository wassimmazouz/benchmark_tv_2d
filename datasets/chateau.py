from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import deepinv as dinv
    import torch
    from benchmark_utils.deepinv_funcs import DeepInverseOperator
    from benchmark_utils.shared import rgb_to_grayscale


class Dataset(BaseDataset):

    name = "chateau"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'type_A': ['denoising', 'inpainting']
    }

    def get_data(self):
        if torch.cuda.is_available():
            device = dinv.utils.get_freer_gpu()
        else:
            device = 'cpu'

        url = (
            "https://culturezvous.com/wp-content/uploads/2017/10/"
            "chateau-azay-le-rideau.jpg?download=true"
        )
        x = dinv.utils.load_url_image(url=url, img_size=100).to(device)
        x_gray = rgb_to_grayscale(x.squeeze(0))
        op = DeepInverseOperator(tensor_size=x_gray.shape, type_A=self.type_A)
        y = op.physics(x_gray.unsqueeze(0))
        return dict(A=op, y=y.numpy().squeeze())
