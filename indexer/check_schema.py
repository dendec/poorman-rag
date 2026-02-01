import lancedb
from core.cfg import IndexingConfig

config = IndexingConfig.from_yaml("config_jokes.yaml")
db = lancedb.connect(config.lancedb_uri)
table = db.open_table(config.table_name)
print(f"Table Name: {config.table_name}")
print(f"Rows: {table.count_rows()}")
print(f"Schema: {table.schema}")
