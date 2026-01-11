# TODO: Enable Column Selection in Preprocessing

## Tasks
- [ ] Update `PreprocessingAPI.create_preprocessing_config` to accept optional column lists
- [ ] Modify `PreprocessingAPI.preprocess` to pass column lists to Preprocessor
- [ ] Update `Preprocessor.__init__` to accept column lists
- [ ] Modify `Preprocessor.detect_feature_types` to use provided lists or auto-detect
- [ ] Ensure transformations apply only to specified columns
- [ ] Verify changes work correctly
