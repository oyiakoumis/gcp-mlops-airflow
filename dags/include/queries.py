get_train_data = """
SELECT *
FROM {{ '.'.join([var.value.DATASET, var.value.TABLE_NAME]) }}
WHERE date BETWEEN DATE('{{ macros.ds_add(ds, -14) }}') AND DATE('{{ ds }}')
"""
