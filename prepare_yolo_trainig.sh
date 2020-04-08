#!/bin/sh

echo "Getting images name..."
python generate_train_validation_txt.py
echo "train.txt & validation.txt succesfully generated!"

echo "Getting classes' names..."
python generate_classes_names.py
echo "classes.names succesfully generated!"

echo "Generating labelled_data.data..."
python generate_labelled_data.py
echo "labelled_data.data succesfully generated!"