from thematos.models import TopicModel, LdaModel, TrainConfig, LdaConfig


def gen_model_configs():
    d = TopicModel.generate_config()
    print(d)
    d = LdaModel.generate_config()
    print(d)
    d = TrainConfig.generate_config()
    print(d)
    d = LdaConfig.generate_config()
    print(d)


if __name__ == "__main__":
    gen_model_configs()