from dlgo.data.processor import GoDataProcessor

processor = GoDataProcessor
features, labels = processor.load_go_data('train', 100)

print(features[0], labels[0])