"""
Jesse Wilson (jesse.wilson@colostate.edu) Colorado State University
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os.path
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
from PIL import Image
from data.base_dataset import BaseDataset, get_params, get_transform


class Rcm2shgDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(dataroot='./dataset/rcm2shg/')
        parser.set_defaults(preprocess_mode='resize_and_crop')
        load_size = 286 if is_train else 256
        parser.set_defaults(load_size=load_size)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=1)
        parser.set_defaults(image_nc=1)
        parser.set_defaults(input_nc=1)
        parser.set_defaults(output_nc=1)
        parser.set_defaults(contain_dontcare_label=False)
        parser.set_defaults(no_instance=True)
        parser.set_defaults(no_input_semantics=True)
        parser.set_defaults(use_wandb=False)
        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        phase = opt.phase

        image_dir = os.path.join(root, '%s' % phase)
        image_paths = make_dataset(image_dir, recursive=False, read_cache=True)

        instance_paths = []
        label_paths = image_paths

        return label_paths, image_paths, instance_paths

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image_both = Image.open(image_path)
        image_both = image_both.convert('RGB') # convert to grayscale, single channel
        size = image_both.size
        
        # Label Image (rcm)
        label = image_both.crop((0,0,size[0]//2,size[1]))

        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, is_image=True)
        label_tensor = transform_label(label)[0:1,:]
        
        # input image (shg)
        image = image_both.crop((size[0]//2, 0, size[0], size[1]))

        transform_image = get_transform(self.opt, params, is_image=True)
        image_tensor = transform_image(image)[0:1,:]

        instance_tensor = 0

        input_dict = {
            'label': label_tensor,
            'instance': instance_tensor,
            'image': image_tensor,
            'path': image_path,
        }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict
