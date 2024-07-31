from preprocess import PreprocessorPyramid, PreprocessorFCU


if __name__ == '__main__':
    preprocessor = PreprocessorFCU()
    preprocessor.load('data', interval=[102,106], window_size=12)
    preprocessor.dump('data/processedFCU_max')

    print(f"Complete preprocessing for FCU data with {preprocessor.n_entries} entries.")


    # preprocessor = PreprocessorPyramid()
    # preprocessor.load('data', interval=[102,106])
    # preprocessor.dump('data/processedPyramid_ws_6')

    # print(f"Complete preprocessing for Pyramid data with {preprocessor.n_entries} entries.")
    