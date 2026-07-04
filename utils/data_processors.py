from sklearn.model_selection import train_test_split

def prepare_tabfm_context(X_train, y_train, max_rows=500, random_state=42):
    """
    Subsamples a representative context dataset from the training set
    to keep it within TabFM's attention scaling limits, preserving target distribution.
    """
    if len(X_train) <= max_rows:
        return X_train, y_train
    
    # Stratified split to subsample max_rows
    # Since we want context size = max_rows, test_size is max_rows / len(X_train)
    # inside train_test_split, we can use test_size to get exactly max_rows in the split.
    test_size = max_rows / len(X_train)
    _, X_context, _, y_context = train_test_split(
        X_train, y_train, 
        test_size=test_size, 
        stratify=y_train, 
        random_state=random_state
    )
    return X_context, y_context
