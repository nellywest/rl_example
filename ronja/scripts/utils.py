import yaml


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return f"{agent_id}_policy"