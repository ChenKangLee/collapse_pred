from preprocess.pyramid import PreprocessorPyramid


if __name__ == '__main__':
    preprocessor = PreprocessorPyramid()
    preprocessor.load('data', interval=[102,106], window_size=6)
    preprocessor.dump('data/processedPyramid_ws_6')

    print(f"Complete preprocessing {preprocessor.n_entries} entries.")
    