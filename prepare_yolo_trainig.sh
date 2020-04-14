#!/bin/sh

echo "Generating complementary text files..."
python ./python/add_txt_classes.py
echo "Text files generated!"

echo "Getting images name..."
python ./python/generate_train_validation_txt.py
echo "train.txt & validation.txt succesfully generated!"

echo "Getting classes' names..."
python ./python/generate_classes_names.py
echo "classes.names succesfully generated!"

echo "Generating labelled_data.data..."
python ./python/generate_labelled_data.py
echo "labelled_data.data succesfully generated!"