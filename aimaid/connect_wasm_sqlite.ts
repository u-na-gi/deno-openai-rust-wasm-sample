import { DB } from "https://deno.land/x/sqlite/mod.ts";

const db = new DB("test.db");

db.close();
