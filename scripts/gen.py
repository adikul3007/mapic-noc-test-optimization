import random

def generate_architectures(n, num_archs=10):
    arch_names = []  # To store unique architecture names
    architectures = []  # List to store the final output
    
    # Function to generate a random architecture name
    def generate_arch_name():
        name = random.choice("abcdefghijklmnopqrstuvwxyz") + str(random.randint(100000, 999999))
        while name in arch_names:  # Ensure uniqueness
            name = random.choice("abcdefghijklmnopqrstuvwxyz") + str(random.randint(100000, 999999))
        arch_names.append(name)
        return name

    # Generate the architectures
    for arch_id in range(1, num_archs+1):
        arch_name = generate_arch_name()
        for core_idx in range(1, n+1):
            # Generate random number of patterns
            if core_idx == 1:
                num_patterns = random.randint(500, 1500)
            else:
                num_patterns = random.randint(1000, 6000) * core_idx // random.randint(1, 3)
            
            # Generate scan chain length based on number of patterns (with some noise)
            scan_chain_len = random.randint(5, 35) + core_idx + num_patterns // random.randint(100, 500)
            
            architectures.append(f"{arch_id} {arch_name} {core_idx} {num_patterns} {scan_chain_len}")
    
    return architectures

# Specify the number of cores per architecture (n=7)
n = 7

# Generate 10 architectures
architectures = generate_architectures(n, num_archs=10)

# Print the generated architectures
for arch in architectures:
    print(arch)
