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
const minimist = require("minimist");

var knownOptions = {
  string: ["input", "output"],
  default: {
    input: "models/compressor/generated",
    output: "assets",
  },
};

var options = minimist(process.argv.slice(2), knownOptions);

lzCompress = function () {
  return through.obj(function (file, enc, cb) {
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
        centroids: centroids.toString(),
      })
    )
    .pipe(rename("word-embeddings.json"))
    .pipe(gulp.dest(options.output));
}

function getWebpackStream(mode) {
  return webpack({
    mode: mode,
    entry: {
      embeddings: "./src/embeddings.js",
    },
    output: {
      filename: "[name].js",
      library: "embeddings",
    },
  });
}

function build() {
  return gulp
    .src("src/embeddings.js")
    .pipe(getWebpackStream("production"))
    .pipe(gulp.dest("assets/"));
}

function copyWasmBinaries() {
  return gulp
    .src("node_modules/@tensorflow/tfjs-backend-wasm/dist/*.wasm")
    .pipe(gulp.dest("assets/"));
}

function copyLstmModel() {
  return gulp
    .src("models/sentiment_lstm/generated/*")
    .pipe(gulp.dest("assets/sentiment_lstm/"));
}

function watchAndBuild() {
  return gulp.watch("src").on("change", function () {
    gulp
      .src("src/embeddings.js")
      .pipe(getWebpackStream("development"))
      .pipe(gulp.dest("assets/"))
      .pipe(connect.reload());
  });
}

function runServer() {
  return connect.server({
    livereload: true,
    middleware: function (connect, opt) {
      return [
        function (req, res, next) {
          if (req.url.endsWith(".wasm")) {
            console.log(req.url);
            res.setHeader("Content-Type", "application/wasm");
          }
          next();
        },
      ];
    },
  });
}

gulp.task(
  "build",
  gulp.parallel(
    gulp.series(compress, buildModel),
    copyWasmBinaries,
    copyLstmModel,
    build
  )
);

gulp.task(
  "build-embeddings",
  gulp.series(compress, buildModel, copyWasmBinaries, build)
);

gulp.task("watch", gulp.parallel(runServer, watchAndBuild));
