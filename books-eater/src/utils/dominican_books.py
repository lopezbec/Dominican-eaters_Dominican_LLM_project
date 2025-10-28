"""Dataset of important Dominican literature books."""

from typing import List, Tuple

# Format: (Título, Autor, Año)
DOMINICAN_BOOKS: List[Tuple[str, str, str]] = [
    # JUAN BOSCH (1909-2001)
    ("La Mañosa", "Juan Bosch", "1936"),
    ("El Oro y la Paz", "Juan Bosch", "1975"),
    ("Camino Real", "Juan Bosch", "1933"),
    ("Dos Pesos de Agua", "Juan Bosch", "N/A"),
    ("Indios", "Juan Bosch", "N/A"),
    ("La Bella Alma de Don Damián", "Juan Bosch", "N/A"),
    ("Hostos el Sembrador", "Juan Bosch", "1939"),
    ("De Cristóbal Colón a Fidel Castro", "Juan Bosch", "1970"),
    ("Judas Iscariote", "Juan Bosch", "N/A"),
    ("Cuentos Más que Completos", "Juan Bosch", "N/A"),
    
    # PEDRO MIR (1913-2000)
    ("Hay un País en el Mundo", "Pedro Mir", "1949"),
    ("Contracanto a Walt Whitman", "Pedro Mir", "1952"),
    ("Amén de Mariposas", "Pedro Mir", "1969"),
    ("La Tercera Isla", "Pedro Mir", "N/A"),
    ("Viaje a la Semilla de la Palabra", "Pedro Mir", "1960"),
    ("Apocalipsis Íntimo", "Pedro Mir", "N/A"),
    ("Celebración de la Alquimia", "Pedro Mir", "N/A"),
    ("La Torre del Silencio", "Pedro Mir", "N/A"),
    
    # SALOMÉ UREÑA (1850-1897)
    ("Poesías Completas", "Salomé Ureña", "1880"),
    ("A la Educación", "Salomé Ureña", "N/A"),
    ("A la Patria", "Salomé Ureña", "N/A"),
    ("La Gloria del Progreso", "Salomé Ureña", "N/A"),
    ("Sueños y Realidades", "Salomé Ureña", "N/A"),
    
    # MANUEL DE JESÚS GALVÁN (1834-1910)
    ("Enriquillo", "Manuel de Jesús Galván", "1882"),
    
    # MANUEL DEL CABRAL (1907-1999)
    ("Viaje Iluminado", "Manuel del Cabral", "1938"),
    ("Compadre Mon", "Manuel del Cabral", "N/A"),
    ("El Presidente Fantasma", "Manuel del Cabral", "N/A"),
    ("Poemas de la Sed", "Manuel del Cabral", "N/A"),
    
    # AIDA CARTAGENA PORTALATÍN (1918-1994)
    ("Escalera para Electra", "Aida Cartagena Portalatín", "1970"),
    ("Tablero", "Aida Cartagena Portalatín", "1954"),
    
    # HILMA CONTRERAS (1907-1996)
    ("Entre Dos Silencios", "Hilma Contreras", "1987"),
    
    # MARCIO VELOZ MAGGIOLO (1936-)
    ("De Abril en Adelante", "Marcio Veloz Maggiolo", "1975"),
    ("El Buen Ladrón", "Marcio Veloz Maggiolo", "1997"),
    
    # RAMÓN MARRERO ARISTY (1913-1959)
    ("Over", "Ramón Marrero Aristy", "1939"),
    
    # FRANKLIN MIESES BURGOS (1907-1976)
    ("Sin Mundo ya y Herido por el Cielo", "Franklin Mieses Burgos", "1944"),
    
    # JULIA ÁLVAREZ (1950-)
    ("En el Tiempo de las Mariposas", "Julia Álvarez", "1994"),
    ("De Cómo las Muchachas García Perdieron el Acento", "Julia Álvarez", "1991"),
    
    # JUNOT DÍAZ (1968-)
    ("La Breve y Maravillosa Vida de Óscar Wao", "Junot Díaz", "2007"),
    ("Así es como la Pierdes", "Junot Díaz", "2012"),
    
    # OTROS AUTORES DESTACADOS
    ("La Sangre", "Pedro Antonio Valdez", "2000"),
    ("El Masacre se Pasa a Pie", "Freddy Prestol Castillo", "1973"),
    ("Apenas un Bolero", "Pedro Antonio Valdez", "1996"),
    ("Curriculum (El Síndrome de la Visa)", "Pedro Antonio Valdez", "2014"),
]


def get_books_as_objects():
    """
    Convert the book tuples to Book objects.
    
    Returns:
        List of Book objects
    """
    from ..models.book import Book
    
    books = []
    for idx, (titulo, autor, año) in enumerate(DOMINICAN_BOOKS, 1):
        books.append(Book(
            numero=idx,
            titulo=titulo,
            autor=autor,
            año=año
        ))
    
    return books
