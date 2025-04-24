from tinydb import TinyDB, Query
from typing import Optional, List, Dict, Any

class TinyDB_manager:
    def __init__(self, db_path: str = "results.json"):
        self.db = TinyDB(db_path)
        self.Result = Query()

    def get_all_ids(self) -> List[Any]:
        """Возвращает список всех уникальных id"""
        return list({entry["id"] for entry in self.db.all() if "id" in entry})

    def save_result(self, record: Dict[str, Any], search_id: Optional[Any] = None) -> None:
        """
        Сохраняет результат в базу. Принимает готовый словарь записи.
        Обязательно: в словаре должен быть ключ 'id'.
        """
        if "id" not in record:
            raise ValueError("Поле 'id' обязательно в record")
        key = search_id if search_id is not None else record["id"]
        self.db.upsert(record, self.Result.id == key)

    def get_id(self, id: Any) -> Optional[Dict[str, Any]]:
        """Возвращает запись по id (в виде словаря), если она существует"""
        result = self.db.search(self.Result.id == id)
        return result[0] if result else None

    def delete_id(self, id: Any) -> bool:
        """Удаляет запись по id. Возвращает True, если запись была удалена"""
        removed = self.db.remove(self.Result.id == id)
        return bool(removed)

    def count(self) -> int:
        """Возвращает количество записей в базе"""
        return len(self.db)

    def export_json(self) -> List[Dict[str, Any]]:
        """Возвращает все записи как список словарей (например, для экспорта или анализа)"""
        return self.db.all()
