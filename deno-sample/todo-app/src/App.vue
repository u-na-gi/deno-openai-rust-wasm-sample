<template>
  <div id="app">
    <h1>Todo App</h1>
    <div class="todo-container">
      <div class="task-list">
        <h2>Tasks</h2>
        <div
          v-for="(task, index) in tasks"
          :key="task.id"
          class="task"
          @click="moveToDone(index)"
        >
          {{ task.text }}
          <button @click.stop="removeTask(index)">Delete</button>
        </div>
      </div>
      <div class="done-list">
        <h2>Done</h2>
        <div
          v-for="(doneTask, index) in doneTasks"
          :key="doneTask.id"
          class="done-task"
        >
          {{ doneTask.text }}
        </div>
      </div>
    </div>
    <input v-model="newTask" placeholder="Add a new task" />
    <button @click="addTask">Add Task</button>
  </div>
</template>

<script>
export default {
  data() {
    return {
      tasks: [],
      doneTasks: [],
      newTask: '',
    };
  },
  methods: {
    addTask() {
      if (this.newTask.trim()) {
        this.tasks.push({ id: Date.now(), text: this.newTask });
        this.newTask = '';
      }
    },
    removeTask(index) {
      this.tasks.splice(index, 1);
    },
    moveToDone(index) {
      const task = this.tasks.splice(index, 1)[0];
      this.doneTasks.push(task);
    }
  }
};
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  text-align: center;
  margin: 20px;
}

.todo-container {
  display: flex;
  justify-content: space-between;
}

.task-list, .done-list {
  width: 45%;
  border: 1px solid #ccc;
  min-height: 200px;
  padding: 10px;
}

.task, .done-task {
  background: #f9f9f9;
  padding: 10px;
  margin: 5px 0;
  cursor: pointer;
}
</style>