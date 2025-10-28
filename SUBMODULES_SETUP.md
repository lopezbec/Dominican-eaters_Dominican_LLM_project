# Guía de Configuración de Submodules

## Estado Actual

Los tres proyectos (`books-eater`, `poems-eater`, `lyrics-eater`) están actualmente dentro del repositorio padre como directorios normales con sus propios repositorios Git.

## Pasos para Convertirlos en Submodules

### Opción 1: Usando Repositorios GitHub Existentes

Si ya tienes los repositorios en GitHub para cada proyecto:

```bash
cd /home/manuujrodcruz/workspace/Dominican-eaters_Dominican_LLM_project

# Remover los directorios actuales (asegúrate de hacer backup primero)
git rm -r books-eater poems-eater lyrics-eater

# Agregar como submodules
git submodule add git@github.com:USUARIO/books-eater.git books-eater
git submodule add git@github.com:USUARIO/poems-eater.git poems-eater
git submodule add git@github.com:USUARIO/lyrics-eater.git lyrics-eater

# Commit los cambios
git commit -m "Add projects as submodules"
git push origin main
```

### Opción 2: Crear Nuevos Repositorios en GitHub

1. **Crear repositorios en GitHub** para cada proyecto:
   - `books-eater`
   - `poems-eater`
   - `lyrics-eater`

2. **Push cada proyecto a su repositorio**:

```bash
# Books Eater
cd books-eater
git remote add origin git@github.com:USUARIO/books-eater.git
git branch -M main
git push -u origin main
cd ..

# Poems Eater
cd poems-eater
git remote add origin git@github.com:USUARIO/poems-eater.git
git branch -M main
git push -u origin main
cd ..

# Lyrics Eater
cd lyrics-eater
git remote add origin git@github.com:USUARIO/lyrics-eater.git
git branch -M main
git push -u origin main
cd ..
```

3. **Convertir a submodules** (desde el directorio padre):

```bash
# Hacer backup de los directorios
mv books-eater ../books-eater-backup
mv poems-eater ../poems-eater-backup
mv lyrics-eater ../lyrics-eater-backup

# Agregar como submodules
git submodule add git@github.com:USUARIO/books-eater.git books-eater
git submodule add git@github.com:USUARIO/poems-eater.git poems-eater
git submodule add git@github.com:USUARIO/lyrics-eater.git lyrics-eater

# Commit y push
git add .gitmodules books-eater poems-eater lyrics-eater
git commit -m "Add projects as Git submodules"
git push origin main

# Verificar
git submodule status
```

### Opción 3: Mantener como Directorios Normales (Más Simple)

Si prefieres no usar submodules por ahora:

```bash
cd /home/manuujrodcruz/workspace/Dominican-eaters_Dominican_LLM_project

# Solo agregar todo al repo padre
git add books-eater poems-eater lyrics-eater
git commit -m "Add all projects to parent repository"
git push origin main
```

**Nota**: Esta opción NO usa submodules, todos los archivos estarán en un solo repositorio.

## Verificar Configuración de Submodules

Después de configurar los submodules:

```bash
# Ver estado de submodules
git submodule status

# Ver configuración
cat .gitmodules

# Actualizar todos los submodules
git submodule update --remote --merge

# Clonar repo con submodules
git clone --recurse-submodules git@github.com:lopezbec/Dominican-eaters_Dominican_LLM_project.git
```

## Workflow con Submodules

### Hacer cambios en un submodule:

```bash
# 1. Ir al submodule
cd books-eater

# 2. Hacer cambios y commit
git add .
git commit -m "Update feature X"
git push origin main

# 3. Volver al repo padre y actualizar la referencia
cd ..
git add books-eater
git commit -m "Update books-eater submodule reference"
git push origin main
```

### Actualizar submodules desde upstream:

```bash
# Actualizar todos
git submodule update --remote --merge
git add .
git commit -m "Update all submodules"
git push origin main

# Actualizar uno específico
cd books-eater
git pull origin main
cd ..
git add books-eater
git commit -m "Update books-eater"
git push origin main
```

## Recomendación

Para este proyecto, recomiendo **Opción 2** (crear repos separados y usar submodules) porque:

1. Cada proyecto puede evolucionar independientemente
2. Mejor organización y mantenimiento
3. Permite contribuciones independientes a cada scraper
4. Los datos históricos de Git se mantienen separados
5. Facilita CI/CD por proyecto

## Estado Actual del Repositorio

```
Dominican-eaters_Dominican_LLM_project/
├── .git/                 # Repo padre inicializado
├── .gitignore           # Configurado
├── README.md            # Documentación principal
├── SUBMODULES_SETUP.md  # Este archivo
├── books-eater/         # Proyecto con .git (listo para ser submodule)
├── poems-eater/         # Proyecto con .git (listo para ser submodule)
└── lyrics-eater/        # Proyecto con .git (listo para ser submodule)
```

**Próximo paso**: Decidir qué opción usar y ejecutar los comandos correspondientes.
