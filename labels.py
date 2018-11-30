import io
import os
import csv
import time

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types


def detect_labels(path):
    """Detects labels in all files within the specified directory and
    returns a lis of the results"""
    client = vision.ImageAnnotatorClient()
    all_labels = []

    for filename in os.listdir(path):
        filename_labels = [filename]
        with io.open(path + '/' + filename, 'rb') as image_file:
            content = image_file.read()

        image = vision.types.Image(content=content)

        response = client.label_detection(image=image)
        labels = response.label_annotations

        for label in labels:
            # set threshold for classification
            if label.score > 0.92:
                filename_labels.append(label.description)

        all_labels.append(filename_labels)

    return all_labels

def detect_labels_uri(csv_file):
    """Detects labels in the file located in Google Cloud Storage or on the
    Web. Each uri is on a new row in the csv file."""
    csv_reader = csv.reader(open(csv_file, newline=''))
    all_labels = []

    for uri in csv_reader:
        uri_string = uri[0]
        uri_labels = [uri_string]
        client = vision.ImageAnnotatorClient()
        image = vision.types.Image()
        image.source.image_uri = uri_string

        response = client.label_detection(image=image)
        labels = response.label_annotations

        for label in labels:
            # set threshold for classification
            if label.score > 0.92:
                uri_labels.append(label.description)

        all_labels.append(uri_labels)

    return all_labels

if __name__ == "__main__":
    time_now = str(round(time.time()))

    #Method 1: local analysis
    directory_path = 'resources'
    results_m1 = 'results/m1-' + time_now + '.csv'

    """Write file names and labels to a csv file. The csv file is named
    based on the current unix time. The csv file has each image on a
    separate row, starting with the file name and followed by the
    categories."""
    file_writer = csv.writer(open(results_m1, 'w', newline=''))
    for image in detect_labels(directory_path):
        file_writer.writerow(image)

    print ("local analysis (m1) done! Results are written in " + results_m1)

    # Method 2: uri analysis
    csv_file = 'uri list.csv'
    results_m2 = 'results/m2-' + time_now + '.csv'

    """Write file names and labels to a csv file. The csv file is named
    based on the current unix time. The csv file has each image on a
    separate row, starting with the file name and followed by the
    categories."""
    file_writer = csv.writer(open(results_m2, 'w', newline=''))
    for image in detect_labels_uri(csv_file):
        file_writer.writerow(image)

    print ("uri analysis (m2) done! Results are written in " + results_m2)
