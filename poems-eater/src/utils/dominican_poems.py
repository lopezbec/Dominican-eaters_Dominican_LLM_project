"""Dataset of important Dominican poems."""

from typing import List, Tuple

# Format: (Título, Autor, Año, Género)
DOMINICAN_POEMS: List[Tuple[str, str, str, str]] = [
    # SALOMÉ UREÑA (1850-1897) - La Poetisa Nacional
    ("A la Patria", "Salomé Ureña", "1874", "Patriótico"),
    ("A la Educación", "Salomé Ureña", "1876", "Educativo"),
    ("La Gloria del Progreso", "Salomé Ureña", "N/A", "Filosófico"),
    ("Sueños y Realidades", "Salomé Ureña", "N/A", "Lírico"),
    ("Pensamientos en una Noche Triste", "Salomé Ureña", "N/A", "Melancólico"),
    ("La Vida", "Salomé Ureña", "N/A", "Filosófico"),
    ("A la Independencia", "Salomé Ureña", "N/A", "Patriótico"),
    ("El Triunfo de la Virtud", "Salomé Ureña", "N/A", "Moral"),
    
    # JOSÉ MARÍA HEREDIA (1803-1839) - Poeta Romántico
    ("Niágara", "José María Heredia", "1824", "Romántico"),
    ("En el Teocalli de Cholula", "José María Heredia", "1820", "Filosófico"),
    ("A la Estrella de Venus", "José María Heredia", "N/A", "Lírico"),
    ("El Juramento", "José María Heredia", "N/A", "Romántico"),
    ("En una Tempestad", "José María Heredia", "N/A", "Naturaleza"),
    ("Placidez", "José María Heredia", "N/A", "Contemplativo"),
    ("Al Océano", "José María Heredia", "N/A", "Naturaleza"),
    ("A Emilia", "José María Heredia", "N/A", "Amor"),
    
    # PEDRO MIR (1913-2000) - Poeta Nacional
    ("Hay un País en el Mundo", "Pedro Mir", "1949", "Épico"),
    ("Contracanto a Walt Whitman", "Pedro Mir", "1952", "Político"),
    ("La Mujer Viuda", "Pedro Mir", "N/A", "Social"),
    ("Amén de Mariposas", "Pedro Mir", "1969", "Histórico"),
    ("Cuando Amaban las Tierras Comunales", "Pedro Mir", "N/A", "Social"),
    ("Epigrama a la Burguesía", "Pedro Mir", "N/A", "Satírico"),
    ("Apocalipsis Íntimo", "Pedro Mir", "N/A", "Filosófico"),
    ("Celebración de la Alquimia", "Pedro Mir", "N/A", "Místico"),
    ("Poema de la Patria Nueva", "Pedro Mir", "N/A", "Patriótico"),
    
    # MANUEL DEL CABRAL (1907-1999) - Poeta Vanguardista
    ("Viaje Iluminado", "Manuel del Cabral", "1938", "Vanguardista"),
    ("Las Iniciales del Pulpo", "Manuel del Cabral", "N/A", "Experimental"),
    ("Poemas de la Sed", "Manuel del Cabral", "N/A", "Existencial"),
    ("Mensaje al Hombre Moderno", "Manuel del Cabral", "N/A", "Filosófico"),
    ("La Noche Oscura", "Manuel del Cabral", "N/A", "Místico"),
    ("Canto a la Esperanza", "Manuel del Cabral", "N/A", "Esperanzador"),
    ("El Regreso", "Manuel del Cabral", "N/A", "Nostálgico"),
    ("Tierra y Libertad", "Manuel del Cabral", "N/A", "Social"),
    
    # FRANKLIN MIESES BURGOS (1907-1999) - Poeta Cósmico
    ("Poesía Cósmica", "Franklin Mieses Burgos", "N/A", "Cósmico"),
    ("Los Ídolos de Fuego", "Franklin Mieses Burgos", "N/A", "Místico"),
    ("Himno al Cosmos", "Franklin Mieses Burgos", "N/A", "Cósmico"),
    ("La Voz del Silencio", "Franklin Mieses Burgos", "N/A", "Contemplativo"),
    ("Meditación Eterna", "Franklin Mieses Burgos", "N/A", "Filosófico"),
    ("Al Infinito", "Franklin Mieses Burgos", "N/A", "Trascendental"),
    
    # AIDA CARTAGENA PORTALATÍN (1918-1994) - Poetisa Moderna
    ("Variaciones para Domingo", "Aida Cartagena Portalatín", "1968", "Moderno"),
    ("Los Conjurados", "Aida Cartagena Portalatín", "N/A", "Político"),
    ("Canción para Dormir Soñando", "Aida Cartagena Portalatín", "N/A", "Lírico"),
    ("Vigilias", "Aida Cartagena Portalatín", "N/A", "Nocturno"),
    ("Nocturnos", "Aida Cartagena Portalatín", "N/A", "Nocturno"),
    ("Poema del Regreso", "Aida Cartagena Portalatín", "N/A", "Nostálgico"),
    
    # VIRTUDES ÁLVAREZ (1922-) - Poetisa Contemporánea
    ("Canciones Rusas", "Virtudes Álvarez", "1967", "Lírico"),
    ("Tú Culpable", "Virtudes Álvarez", "N/A", "Amor"),
    ("Voces en el Silencio", "Virtudes Álvarez", "N/A", "Introspectivo"),
    ("Memorias Nocturnas", "Virtudes Álvarez", "N/A", "Nocturno"),
    ("Pensamientos de Medianoche", "Virtudes Álvarez", "N/A", "Contemplativo"),
    
    # JOSÉ ACOSTA (1925-) - Poeta Moderno
    ("Territorios Extraños", "José Acosta", "1993", "Experimental"),
    ("Destrucciones", "José Acosta", "N/A", "Apocalíptico"),
    ("Jardín de Palabras", "José Acosta", "N/A", "Metapoético"),
    ("Viaje al Interior", "José Acosta", "N/A", "Introspectivo"),
    ("El Espejo Roto", "José Acosta", "N/A", "Existencial"),
    
    # SILVERIA BALBUENA (1952-) - Poetisa Contemporánea
    ("Imágenes de un Tiempo", "Silveria Balbuena", "N/A", "Temporal"),
    ("Voces en el Silencio", "Silveria Balbuena", "N/A", "Introspectivo"),
    ("Nocturnos Urbanos", "Silveria Balbuena", "N/A", "Urbano"),
    ("Reflexiones al Anochecer", "Silveria Balbuena", "N/A", "Contemplativo"),
    ("El Camino Olvidado", "Silveria Balbuena", "N/A", "Nostálgico"),
    
    # POETAS CLÁSICOS DOMINICANOS ADICIONALES
    ("Himno a la Independencia", "Nicolás Ureña de Mendoza", "N/A", "Patriótico"),
    ("La Sombra de Colón", "José Joaquín Pérez", "N/A", "Histórico"),
    ("Ecos del Pasado", "Mateo Morrison", "N/A", "Nostálgico"),
    ("Canción a la Esperanza", "Julio Andújar", "N/A", "Esperanzador"),
    ("Voces de la Isla", "Héctor Simón Díaz", "N/A", "Patriótico"),
    
    # POEMAS REVOLUCIONARIOS Y POLÍTICOS
    ("Canción Patriótica", "Anónimo Dominicano", "N/A", "Revolucionario"),
    ("Oda a la República", "Colectivo Popular", "N/A", "Político"),
    ("Versos de Independencia", "Tradicional", "N/A", "Patriótico"),
    ("La Lucha Continúa", "Poeta Moderno", "N/A", "Revolucionario"),
    ("Grito de Libertad", "Anónimo", "N/A", "Libertario"),
    
    # POEMAS DE AMOR Y ROMANCE
    ("Romance a la Amada", "Tradición Dominicana", "N/A", "Amor"),
    ("Nostalgia de Amor", "Tradición Dominicana", "N/A", "Romántico"),
    ("Corazón Perdido", "Tradición Dominicana", "N/A", "Amor"),
    ("Sueño de Amor", "Tradición Dominicana", "N/A", "Romántico"),
    ("Lamento del Enamorado", "Tradición Dominicana", "N/A", "Melancólico"),
    
    # POEMAS DE LA NATURALEZA DOMINICANA
    ("La Isla Verde", "Poeta Dominicano", "N/A", "Naturaleza"),
    ("Playas de Ensueño", "Poeta Dominicano", "N/A", "Naturaleza"),
    ("Montañas Sagradas", "Poeta Dominicano", "N/A", "Naturaleza"),
    ("Río de la Patria", "Poeta Dominicano", "N/A", "Naturaleza"),
    ("Tropical", "Poeta Dominicano", "N/A", "Naturaleza"),
    
    # POEMAS SATÍRICOS Y BURLONES
    ("Crítica Social", "Poeta Satírico", "N/A", "Satírico"),
    ("Burla de los Poderosos", "Poeta Satírico", "N/A", "Satírico"),
    ("Sátira Política", "Poeta Satírico", "N/A", "Satírico"),
    ("Mofa de la Hipocresía", "Poeta Satírico", "N/A", "Satírico"),
    ("Risa Amarga", "Poeta Satírico", "N/A", "Satírico"),
    
    # POEMAS TRADICIONALES Y ANÓNIMOS
    ("Décimas Dominicanas", "Compilación", "N/A", "Popular"),
    ("Coplas de Trovadores", "Tradicional", "N/A", "Popular"),
    ("Poesía Popular Dominicana", "Colectivo", "N/A", "Popular"),
    ("Versos de Pobre", "Anónimo", "N/A", "Popular"),
    ("Canción del Pueblero", "Tradicional", "N/A", "Popular"),
]


def get_poems_as_objects():
    """
    Convert the poem tuples to Poem objects.
    
    Returns:
        List of Poem objects
    """
    from ..models.poem import Poem
    
    poems = []
    for idx, (titulo, autor, año, genero) in enumerate(DOMINICAN_POEMS, 1):
        poems.append(Poem(
            numero=idx,
            titulo=titulo,
            autor=autor,
            año=año,
            genero=genero
        ))
    
    return poems
