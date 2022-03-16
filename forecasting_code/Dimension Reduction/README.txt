USAGE: 

1) Set the current working directory to ``~/.../Dimension Reduction''
2) Set settings for a read/generation script in the corresponding job file.
3) Run a read/generation script (e.g ``python ./generate_data.py'') 
4) Set settings for dim_reduce.py/ dim_optimality.py in the corresponding job file (dim_optimality.py uses settings from both dim_optimality_job.py and dim_reduce_job.py; all training params are assumed fixed as they exist in dim_reduce_job.py by dim_optimality.py except for 'm', at this point).
5) Run dim_reduce.py or dim_optimality.py (e.g ``python ./dim_reduce.py'') 

