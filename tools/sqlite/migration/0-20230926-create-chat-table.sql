

CREATE TABLE IF NOT EXISTS chat {
  id          INTEGER PRIMARY KEY NOT NULL,
  thread_id INTEGER NOT NULL,
  user TEXT NOT NULL,
  content TEXT NOT NULL,
}

CREATE TABLE IF NOT EXISTS thread {
  id INTEGER PRIMARY KEY NOT NULL,
  title TEXT NOT NULL,
}

CREATE UNIQUE INDEX thread_title_index on thread(title);
