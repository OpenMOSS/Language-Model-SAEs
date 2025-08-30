cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate 

db.analyses.deleteOne({"sae_name":"lc0-L0-20-exp16-initmode"})
db.analyses.deleteOne({"sae_name":"lc0-L7-20-exp16-initmode"})
db.analyses.deleteOne({"sae_name":"lc0-L14-20-exp16-initmode"})

db.features.deleteMany({"sae_name":"lc0-L0-20-exp16-initmode"})
db.features.deleteMany({"sae_name":"lc0-L7-20-exp16-initmode"})
db.features.deleteMany({"sae_name":"lc0-L14-20-exp16-initmode"})