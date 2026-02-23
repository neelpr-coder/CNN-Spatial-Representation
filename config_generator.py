import os
import yaml
import utils
import data


def generate_config(
        template_config,
        env_nums,
        n_rotations_list,
        movement_modes,
        model_names,
    ):
    template = utils.load_config(template_config)
    for env_num in env_nums:
        for n_rotations in n_rotations_list:
            for movement_mode in movement_modes:
                for model_name in model_names:
                    output_layers = data.load_model_layers(model_name)
                    for output_layer in output_layers:
                        config_version = \
                            f'env{env_num}_r{n_rotations}_{movement_mode}_' \
                            f'{model_name}_{output_layer}'
                        
                        template['config_version'] = config_version
                        template['unity_env'] = f'env{env_num}_r{n_rotations}'
                        template['n_rotations'] = n_rotations
                        template['movement_mode'] = movement_mode
                        template['model_name'] = model_name
                        template['output_layer'] = output_layer
                        
                        base_dir = os.path.dirname(os.path.abspath(__file__))  # Space-units/
                        configs_dir = os.path.join(base_dir, 'configs')
                        os.makedirs(configs_dir, exist_ok=True)
                        with open(os.path.join(configs_dir, f'{config_version}.yaml'), 'w') as f:
                            yaml.dump(template, f, sort_keys=False)


if __name__ == "__main__":
    generate_config(
        template_config='env17_r24_2d_vgg16_fc2',
        env_nums=['17run2'],
        n_rotations_list=[24],
        movement_modes=['2d'],
        model_names=['vgg16'],
    )