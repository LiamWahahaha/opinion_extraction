<template>
  <v-form>
    <v-textarea v-model="content" filled label="Content" auto-grow></v-textarea>
    <v-spacer />
    <v-btn @click="clear">clear</v-btn>
    <v-btn @click="submit" class="mr-4">extract opinion</v-btn>
    <v-textarea v-model="result" filled label="Opinion" auto-grow></v-textarea>
  </v-form>
</template>

<script>
import axios from 'axios'

export default {
  data: () => ({
    content: '',
    result: '',
    error: ''
  }),

  computed: {},

  methods: {
    submit() {
      axios
        .post(`http://127.0.0.1:5000/extract-opinion`, {
          msg: this.content
        })
        .then((response) => {
          this.result = JSON.stringify(response.data)
        })
        .catch((error) => {
          this.error = JSON.stringify(error.data)
        })
    },
    clear() {
      this.content = ''
      this.result = ''
      this.error = ''
    }
  }
}
</script>
