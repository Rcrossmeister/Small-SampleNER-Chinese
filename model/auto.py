import os

for i in range(5, 6):
    for j in range(9, 0, -1):
        # with open("config.toml", "w", encoding="utf-8") as file:
        #     file.write(f'[config]\n\nlearning_rate = {j}e-{i}\nbatch_size = 32\noptimizer = "AdamW"\nepochs = 200\n#log_level = "DEBUG"\nlog_level = "INFO"\n')
        os.system(f"sed -i /learning_rate/s/[0-9]e-[0-9]/{j}e-{i}/ ./config.toml")
        os.system(f"$(which python) main.py > train[{j}e-{i}].log 2>&1")

# sed -i "/learning_rate/s/[0-9]e-[0-9]/2e-6/" ./mmt/framework/config.toml

# for j in ["2e-4", "1e-4", "9e-5"]:
#     os.system(f"sed -i /learning_rate/s/[0-9]e-[0-9]/{j}/ ./config.toml")
#     os.system("$(which python) main.py")

