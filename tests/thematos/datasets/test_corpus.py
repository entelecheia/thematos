from thematos.datasets.corpus import Corpus


def test_corpus():
    d = Corpus.generate_config()
    print(d)


if __name__ == "__main__":
    test_corpus()
