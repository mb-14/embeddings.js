const gulp = require('gulp');
const through = require('through2');
const lzstring = require('lz-string');
const Vinyl = require('vinyl');
const path = require('path');
const template = require('gulp-template');
const fs = require('fs');
const rename = require('gulp-rename');
const webpack = require('webpack-stream');
const del = require('del');

lzCompress = function() {
  return through.obj(function(file, enc, cb){
   var contents = file.contents.toString();
   var base = path.join(file.path, '..');
   var compressed = lzstring.compressToBase64(contents);
   var compressedFile = file.clone();
   compressedFile.contents = new Buffer(compressed);
   compressedFile.basename += '.lz'
   cb(null, compressedFile);
 });
};


function clean() {
  return del(['dist', 'demo/assets', 'src/model.js', 'model/codes.json.lz', 'model/centroids.json.lz']);
}

function compress() {
  return gulp.src(['model/codes.json', 'model/centroids.json'])
  .pipe(lzCompress())
  .pipe(gulp.dest('model'));
}


function buildModel() {
  var vocabulary = fs.readFileSync('model/vocab.json');
  var codes = fs.readFileSync('model/codes.json.lz');
  var centroids = fs.readFileSync('model/centroids.json.lz');
  return gulp.src('src/model.tmpl.json')
  .pipe(template({
    vocabulary: vocabulary,
    codes: codes.toString(),
    centroids: centroids.toString() 
  }))
  .pipe(rename('model.json'))
  .pipe(gulp.dest('dist'))
  .pipe(gulp.dest('demo/assets'));
}

function bundle() {
  return gulp.src('src/embeddings.js')
    .pipe(webpack({
      mode: 'development',
      entry: {
        embeddings: './src/embeddings.js',
      },
      output: {
        filename: '[name].js',
        library: 'embeddings'
      }
    }))
    .pipe(gulp.dest('dist/'))
    .pipe(gulp.dest('demo/assets/'));
}

gulp.task('compress', compress);
gulp.task('build-model', buildModel);
gulp.task('clean', clean);
gulp.task('build', gulp.series(clean, compress, buildModel, bundle));