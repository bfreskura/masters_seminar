import data_loader
import constants
import input_process


def main():
    wjs_data = data_loader.parse_WJS(constants.WJS_DATA_DIR)
    input_process.embed_words(wjs_data, embedding="glove")


if __name__ == "__main__":
    main()
