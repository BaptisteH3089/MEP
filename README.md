# MEP

The code of the application montage IA for DAJ.

# Usage of the scripts

## Creation of the database of a customer

It's not a relational database, it is just an ensemble of files that will be used to learn some artificial intelligence models and to compute proposals of pages.
The different files of the database are:
- dict_pages: Dictionary python with all the information we have about the pages of that customer.
- dict_layouts_small: A dictionary with the layouts frequently used and with a MelodyId.
- dict_page_array: A dictionary with all the layouts (with or without MelodyId).
- dict_page_array_fast: Very similar to dict_page_array but allow to do faster comparisons between layouts with the same number of modules.
- list_mdp: A list that gather all the MelodyIds of pages that used the same layouts.

### Structure of the objects in the database

- dict_pages = {id_page: dict_page, ...}

The dict_page contains the elements:
- 'pageName': str,
- 'melodyId': float,
- 'cahierCode': str,
- 'customerCode': str,
- 'pressTitleCode': str,
- 'editionCode': str,
- 'folio': int,
- 'width': float,
- 'height': float,
- 'paddingTop': float,
- 'paddingLeft': float,
- 'catName': str,
- 'nbArt': int,
- 'pageTemplate': str,
- 'nameTemplate': str,
- 'propArt': 0.7,
- 'propImg': 0.1,
- 'nbCar': 7200,
- 'nbImg': 4
- 'cartons': list of dict_carton
  - dict_carton contains the elements:
  - 'x': float,
  - 'y': float,
  - 'height': float,
  - 'width': float,
  - 'nbCol': int,
  - 'nbPhoto': int
- 'articles': list of dict_article
  - dict_article contains the elements:
  - 'melodyId': 639397,
  - 'nbCol': 6,
  - 'x': 15.0,
  - 'y': 79.5,
  - 'width': 280.0,
  - 'height': 268.0,
  - 'nbSign': 4172,
  - 'syn': 0,
  - 'supTitle': 0,
  - 'title': 50,
  - 'secTitle': 0,
  - 'subTitle': 265,
  - 'abstract': 0,
  - 'nbSignTxt': 4122,
  - 'content': str,
  - 'titleContent': str,
  - 'nbBlock': 9,
  - 'nbSignBlock': [541, ...],
  - 'typeBlock': ['texte', ...],
  - 'author': 4,
  - 'commune': 0,
  - 'exergue': 0,
  - 'note': 0,
  - 'nbPhoto': 3,
  - 'posArt': 1,
  - 'isPrinc': 1,
  - 'isSec': 0,
  - 'isMinor': 0,
  - 'isSub': 0,
  - 'isTer': 0,
  - 'aire_img': int
  - 'score': 0.8,
  - 'distToPrinc': 1.0,
  - 'aire': 3,
  - 'aireImg': 14400,
  - 'intertitre': 1,
  - 'petittitre': 0,
  - 'quest_rep': 0
  - 'photo': list of dict_photo
  - dict_photo contains:
  - 'x': float,
  - 'y': float,
  - 'xAbs': float,
  - 'yAbs': float,
  - 'width': int,
  - 'height': int,
  - 'credit': int,
  - 'legende': int.

- dict_layouts_small. This dictionary is of the form:
  - {id_layout: {'nameTemplate': str, 'cartons': list_cartons, 'id_pages': list_ids_page, 'array': numpy_array_layout}, ...}
  - numpy_array_layout = numpy.array([[x, y, width, height], ...])
  - The numpy_array_layout are just the most basic layout with just the location of the different modules.

- dict_page_array. This dictionary is of the form:
  - {id_page: numpy_array_layout, ...}
- dict_page_array_fast. This dictionary is of the form:
  - {nb_modules: dict_page_array_filter, ...}
- list_mdp. This list is of the form:
  - [(nb_pages_using_layout, numpy.array(layout), list_ids_page_using_layout), ...]


