from thematos.models import (
    TopicModel,
    LdaModel,
    TrainConfig,
    LdaConfig,
    WordcloudConfig,
)
from thematos.runners import TopicRunner, LdaRunConfig
from thematos.plots.wordcloud import WordCloud


def test_gen_model_configs():
    d = TopicModel.generate_config()
    print(d)
    d = LdaModel.generate_config()
    print(d)
    d = TrainConfig.generate_config()
    print(d)
    d = LdaConfig.generate_config()
    print(d)
    d = TopicRunner.generate_config()
    print(d)
    d = LdaRunConfig.generate_config()
    print(d)
    d = WordCloud.generate_config()
    print(d)
    d = WordcloudConfig.generate_config()
    print(d)


if __name__ == "__main__":
    test_gen_model_configs()
