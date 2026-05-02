from typing import Iterator
from core.datasource import DataSource, DataEntry

class ExampleDataSource(DataSource):
    """
    Template for creating a new Knowledge Base data source.
    """
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def __iter__(self) -> Iterator[DataEntry]:
        # Your logic to fetch and yield data
        items = [{"text": "Hello world", "meta": "greeting"}]
        
        for item in items:
            yield DataEntry(
                embedding_text=item["text"],
                text=item["text"],
                metadata={"source": item["meta"]}
            )

    def __len__(self) -> int:
        return 1
