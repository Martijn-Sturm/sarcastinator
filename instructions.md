requirements2.txt !! some packages changed !!

```bash
cd src
```

check first: 
COMMENTS_FILE = "../data/comments.json"
TRAIN_MAP_FILE = "../data/my_train_balanced.csv"
TEST_MAP_FILE = "../data/my_test_balanced.csv"
if these files are in there, run:
```bash
python process_data.py [path/to/FastText_embedding]
```

Now check:
mainbalancedpickle.p should be in src directory
run: 
```bash
python prepare.py
```

The following folders should be created in src directory:
input_data/train
input_data/test
These should contain "x.p", "y.p", "word_embs.p", and files for author and topic.
Check this.

Then run:
```bash
python run_single.py
```

Gather the logs folder in the src directory, and commit and push this to our repo.

after runs, commit and push logs folder to our repo