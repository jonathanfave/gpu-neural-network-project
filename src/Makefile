# Nom de l'exécutable
TARGET = my_project

# Compilateur CUDA
NVCC = nvcc

# Options de compilation
NVCC_FLAGS = -std=c++11 -O3

# Répertoires d'inclusion
INCLUDES = -I.

# Fichiers sources
SRCS = main.cu ann.cu matrix.cu mnist.cu

# Fichiers objets (générés après compilation)
OBJS = $(SRCS:.cu=.o)

# Règle par défaut pour construire l'exécutable
all: $(TARGET)

# Règle pour créer l'exécutable à partir des objets
$(TARGET): $(OBJS)
	@echo "Building executable $(TARGET)"
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Règle pour compiler les fichiers sources en fichiers objets
%.o: %.cu
	@echo "Compiling $<"
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ -c $<

# Règle pour nettoyer les fichiers générés (objets et exécutable)
clean:
	@echo "Cleaning up..."
	rm -f $(OBJS) $(TARGET)

# Règle pour exécuter le programme (optionnelle)
run: $(TARGET)
	./$(TARGET)
