<?php
$allBahasa = array_values(array_filter(scandir('kamus'), fn ($item) => strpos($item, '.txt') > 0));

function ambilDataset($fileBahasa = null)
{
    $fileBahasa = isset($_GET['bahasa']) ? $_GET['bahasa'] : $fileBahasa;
    if (is_null($fileBahasa)) {
        return [];
    }

    return file_get_contents('kamus/'.$fileBahasa);
}
// echo ambilDataset('jawa.txt');
?>

<html>
    <h1>Heritage Translator (development)</h1>
    Indonesia - 
    <select id="pilihan-bahasa" onchange=changeBahasa()>
        <option value="-">Pilih bahasa</option>
        <?php foreach($allBahasa as $bahasa): ?>
            <option value="<?=$bahasa?>"><?=str_replace('.txt', '', $bahasa)?></option>
        <?php endforeach; ?>
    </select>

    <table>
        <tbody id="table-bahasa"></tbody>
    </table>

    <script>
        function changeBahasa() {
            var select = document.getElementById('pilihan-bahasa')
            window.location.href = 'http://localhost:2000?bahasa=' + select.value
        }
    </script>
</html>