## Introduction 
To develop Custom Machine learning code to perform  text classification on Malaysia news and articles


## define train test split

```
def train_test (df):

	## change the column name below according (df.body = X  , df.sect = Y output)
    x_train, x_test, y_train, y_test = train_test_split( df.body, df.sect, test_size=0.15, random_state=42)
    x_train.index = range(0, len(x_train))
    x_test.index = range(0, len(x_test))
    y_train.index = range(0, len(y_train))
    y_test.index = range(0, len(y_test))
```



