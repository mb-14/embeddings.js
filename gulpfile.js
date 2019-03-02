const gulp = require("gulp");
const through = require("through2");
const lzstring = require("lz-string");
const Vinyl = require("vinyl");
const path = require("path");
const template = require("gulp-template");
const fs = require("fs");
const rename = require("gulp-rename");
const webpack = require("webpack-stream");
const del = require("del");
const connect = require("gulp-connect");
const minimist = require('minimist');

var knownOptions = {
  string: ['input', 'output'],
  default: {
    input: 'models/compressor/generated',
    output: 'pretrained/word-embeddings.json'
  }
};

var options = minimist(process.argv.slice(2), knownOptions);

lzCompress = function() {
  return through.obj(function(file, enc, cb) {
    var contents = file.contents.toString();
    var base = path.join(file.path, "..");
    var compressed = lzstring.compressToBase64(contents);
    var compressedFile = file.clone();
    compressedFile.contents = new Buffer(compressed);
    compressedFile.basename += ".lz";
    cb(null, compressedFile);
  });
};

function compress() {
  return gulp
    .src([`${options.input}/codes.json`, `${options.input}/centroids.json`])
    .pipe(lzCompress())
    .pipe(gulp.dest(`${options.input}/`));
}

function buildModel() {
  var vocabulary = fs.readFileSync(`${options.input}/vocab.json`);
  var codes = fs.readFileSync(`${options.input}/codes.json.lz`);
  var centroids = fs.readFileSync(`${options.input}/centroids.json.lz`);
  return gulp
    .src("src/model.tmpl.json")
    .pipe(
      template({
        vocabulary: vocabulary,
        codes: codes.toString(),
        centroids: centroids.toString()
      })
    )
    .pipe(rename(options.output.substring(options.output.lastIndexOf('/')+1)))
    .pipe(gulp.dest(options.output.substring(0, options.output.lastIndexOf('/'))));
}

function getWebpackStream(mode) {
  return webpack({
    mode: mode,
    entry: {
      embeddings: "./src/embeddings.js"
    },
    externals: {
      tfjs: "tf"
    },
    output: {
      filename: "[name].js",
      library: "embeddings"
    }
  });
}

function build() {
  return gulp
    .src("src/embeddings.js")
    .pipe(getWebpackStream("production"))
    .pipe(gulp.dest("dist/"));
}

function watchAndBuild() {
  return gulp.watch("src").on("change", function() {
    gulp
      .src("src/embeddings.js")
      .pipe(getWebpackStream("development"))
      .pipe(gulp.dest("dist/"))
      .pipe(connect.reload());
  });
}

function runServer() {
  return connect.server({
    livereload: true
  });
}

gulp.task("build-embeddings", gulp.series(compress, buildModel));
gulp.task("build", build);
gulp.task("watch", gulp.parallel(runServer, watchAndBuild));
