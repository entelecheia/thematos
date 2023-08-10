from thematos.models import WordPrior


def test_wordprior():
    d = WordPrior.generate_config()
    print(d)


if __name__ == "__main__":
    test_wordprior()
