from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32

wine_entity = Entity(name="wine_sample", join_keys=["sample_id"], value_type=ValueType.INT64, description="Unique Wine dataset sample")

wine_source = FileSource(name="wine_source", path="data/wine_features.parquet", timestamp_field="event_timestamp")

wine_features = FeatureView(name="wine_measurements", entities=[wine_entity], ttl=timedelta(days=365), schema=[Field(name="alcohol", dtype=Float32), Field(name="malic_acid", dtype=Float32), Field(name="ash", dtype=Float32), Field(name="total_phenols", dtype=Float32), Field(name="flavanoids", dtype=Float32), Field(name="color_intensity", dtype=Float32), Field(name="proline", dtype=Float32), Field(name="phenols_flavanoids_ratio", dtype=Float32), Field(name="color_proline_ratio", dtype=Float32)], source=wine_source)
